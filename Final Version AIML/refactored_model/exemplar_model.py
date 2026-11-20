#!/usr/bin/env python3
"""
exemplar_model.py -- the exemplar-conditioned generator (needs torch; RDKit only for the evaluation helpers).

TWO ARCHITECTURES (same interface: target_logits(S_in, E, roles) -> next-token logits):
  ExemplarLM    (default, FORMAT_LM)  decoder-only; the exemplars are written in front of the molecule in ONE causal sequence.
  ExemplarModel (FORMAT)              the encoder + cross-attention design described below; in tests it did not learn to use its exemplars (--arch xattn).

ExemplarLM writes the known ligands in front of the molecule as one sequence  [BOS ex1 EOS] ... [BOS exK EOS] [BOS molecule EOS]  and a causal transformer continues it; an optional
pointer (copy) channel lets it copy from the exemplars.  It is the design used by the shipped checkpoints.

ExemplarModel reuses the base model's pieces (SMILES embedding, positional encoding, 3-layer encoder, 3-layer decoder, output
projection) but its decoder cross-attends to the ENCODED EXEMPLAR SMILES instead of disease/target ID embeddings:

    exemplars  [B, K, Le]   token ids of K known ligands (0 = padding; an unused slot is all padding)
    roles      [B, K]       0 none | 1 target-pool exemplar | 2 disease-pool exemplar | 3 (disease, target)-pair exemplar
      -> embed (shared SMILES embedding) + role embedding + positional encoding -> shared encoder, ONE exemplar at a time
      -> memory = [learned null slot] + all exemplar encodings, with a padding mask
    decoder(target SMILES so far, memory)  ->  next-token logits

Each exemplar is encoded on its own (cost grows with K, not K^2) so the model is permutation-invariant over exemplars and
works for any K from 0 to max_exemplars.  The always-present null slot lets a request with no exemplars behave as
unconditional generation.  There are no disease/target ID tables at all, so nothing here depends on a fixed vocabulary.
"""
import copy
import inspect
import math

import numpy as np
import torch
import torch.nn as nn

import exemplar_data as xd
from base_data import DrugDataPreprocessor
from base_model import DrugDesignModel

# format tags stored in checkpoints: FORMAT = ExemplarModel (encoder + cross-attention to the exemplar encodings), FORMAT_LM = ExemplarLM (decoder-only, exemplars written in front of the molecule)
FORMAT = 'exemplar_v1'
FORMAT_LM = 'exemplar_lm_v1'


# Encoder + cross-attention design.  Each exemplar SMILES is encoded on its own (SMILES embedding + role embedding + position); the encodings plus a learned 'null' slot form the
# memory that the pretrained decoder cross-attends to.  It reuses the base model's embedding, decoder and output projection and a copy of its encoder.  In tests it did not learn
# to use its exemplars; ExemplarLM is the design that does.
class ExemplarModel(nn.Module):
    # Takes over the base model's pieces, copies its encoder and adds the role embedding and the null memory slot.  reset_encoder replaces the encoder by a fresh one;
    # reset_cross_attn re-initialises the decoder's cross-attention in place.
    def __init__(self, base, max_exemplars=6, reset_cross_attn=False, reset_encoder=False):
        super().__init__()
        d = base.d_model
        self.d_model = d
        self.nhead = base.nhead
        self.vocab_size = base.vocab_size
        self.max_exemplars = max_exemplars
        self.smiles_embed = base.smiles_embed
        self.pos_encoder = base.pos_encoder
        self.decoder = base.decoder
        self.output_proj = base.output_proj
        self.encoder = copy.deepcopy(base.encoder)
        if reset_encoder:
            # a fresh encoder of the same shape: the pretrained one was trained on ID embeddings, not on SMILES
            l0 = base.encoder.layers[0]
            layer = nn.TransformerEncoderLayer(d, base.nhead, l0.linear1.out_features, l0.dropout.p, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=len(base.encoder.layers))
        if reset_cross_attn:
            # the pretrained cross-attention may have learned that its memory carries little information; a fresh one is loaded in place (self-attention and feed-forward are kept)
            for lyr in self.decoder.layers:
                old = lyr.multihead_attn
                fresh = nn.MultiheadAttention(d, base.nhead, dropout=old.dropout, batch_first=True)
                old.load_state_dict(fresh.state_dict())
        # role embedding (0 none, 1 target pool, 2 disease pool, 3 pair pool); null_mem is the always-present memory slot, so a request with no exemplars still has something to attend to
        self.role_embed = nn.Embedding(4, d)
        nn.init.normal_(self.role_embed.weight, std=0.02)
        self.null_mem = nn.Parameter(torch.zeros(1, 1, d))

    # Encodes every exemplar slot on its own.  E is [B, K, Le] token ids and roles is [B, K].  Returns (memory [B, 1 + K*Le, d], pad_mask [B, 1 + K*Le], True = padding); the first
    # memory position is the learned null slot and is never masked.
    def encode_exemplars(self, E, roles):
        B, K, Le = E.shape
        flat = E.reshape(B * K, Le)
        pad = flat == 0
        empty = pad.all(dim=1)
        enc_pad = pad.clone()
        # an empty slot would be fully masked inside the encoder, which gives NaN; its first position is unmasked (the slot is masked again as memory)
        enc_pad[empty, 0] = False
        x = self.smiles_embed(flat) * math.sqrt(self.d_model)
        x = x + self.role_embed(roles.reshape(B * K)).unsqueeze(1)
        x = self.pos_encoder(x)
        h = self.encoder(x, src_key_padding_mask=enc_pad).reshape(B, K * Le, self.d_model)
        mem_pad = pad.reshape(B, K * Le)
        null = self.null_mem.expand(B, 1, -1).to(h.dtype)
        memory = torch.cat([null, h], dim=1)
        mem_pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=E.device), mem_pad], dim=1)
        return memory, mem_pad

    # Decoder pass: embeds the target tokens so far and runs the decoder with cross-attention to the exemplar memory.  Returns logits [B, L, vocab].
    def forward(self, S_in, E, roles, tgt_mask=None):
        memory, mem_pad = self.encode_exemplars(E, roles)
        tgt = self.pos_encoder(self.smiles_embed(S_in) * math.sqrt(self.d_model))
        out = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_key_padding_mask=mem_pad)
        return self.output_proj(out)

    # Next-token logits at every position of S_in [B, L] (causal mask), shape [B, L, V].  This is the interface that the shared helpers (per_token_nll, sampling) use for both architectures.
    def target_logits(self, S_in, E, roles):
        return self.forward(S_in, E, roles, tgt_mask=causal_mask(S_in.size(1), S_in.device))


# Decoder-only, in-context exemplar model.  The known ligands are written in front of the molecule to be generated as one token sequence, [BOS ex1 EOS] ... [BOS exK EOS]
# [BOS molecule EOS], and a causal transformer continues it (the loss is on the last segment only).  Positions restart at 0 for every molecule, and each token carries a role
# embedding (1 target pool, 2 disease pool, 3 pair pool, 4 the molecule being generated).  K = 0 is the molecule alone.  The layers start from the pretrained decoder's
# self-attention and feed-forward blocks (its cross-attention is dropped), plus extra_layers fresh ones.  With pointer=True a copy channel mixes in a distribution over the
# tokens that follow matching positions of the exemplars.
class ExemplarLM(nn.Module):

    # number of role ids: 0 none, 1 target pool, 2 disease pool, 3 pair pool, 4 the molecule being generated
    N_ROLES = 5
    TARGET_ROLE = 4

    # Builds the blocks: each pretrained decoder layer becomes an encoder-style layer (self-attention + feed-forward, used with a causal mask) with the pretrained weights copied
    # in, followed by extra_layers fresh layers.  It shares the base model's embedding, positional encoding and output projection.  pointer=True adds the copy-channel projections.
    def __init__(self, base, max_exemplars=6, extra_layers=0, pointer=False):
        super().__init__()
        d = base.d_model
        self.d_model = d
        self.nhead = base.nhead
        self.vocab_size = base.vocab_size
        self.max_exemplars = max_exemplars
        self.extra_layers = extra_layers
        self.pointer = pointer
        self.smiles_embed = base.smiles_embed
        self.pos_encoder = base.pos_encoder
        self.output_proj = base.output_proj
        blocks = []
        for dl in base.decoder.layers:
            el = nn.TransformerEncoderLayer(d, base.nhead, dl.linear1.out_features, dl.dropout.p, batch_first=True)
            el.self_attn.load_state_dict(dl.self_attn.state_dict())
            el.linear1.load_state_dict(dl.linear1.state_dict())
            el.linear2.load_state_dict(dl.linear2.state_dict())
            el.norm1.load_state_dict(dl.norm1.state_dict())
            # in a decoder layer norm1 follows the self-attention and norm3 follows the feed-forward
            el.norm2.load_state_dict(dl.norm3.state_dict())
            blocks.append(el)
        l0 = base.decoder.layers[0]
        for _ in range(extra_layers):
            blocks.append(nn.TransformerEncoderLayer(d, base.nhead, l0.linear1.out_features, l0.dropout.p, batch_first=True))
        self.blocks = nn.ModuleList(blocks)
        self.role_embed = nn.Embedding(self.N_ROLES, d)
        nn.init.normal_(self.role_embed.weight, std=0.1)
        if pointer:
            # copy channel (pointer-generator): the next token is a mixture of the usual vocabulary distribution and 'the token that FOLLOWS the exemplar position whose context best matches
            # mine'.  This gives copying a direct gradient instead of waiting for an attention circuit to form
            self.ptr_q = nn.Linear(d, d)
            self.ptr_k = nn.Linear(d, d)
            self.ptr_gate = nn.Linear(d, 1)

    # Lays the exemplars and the target out as ONE sequence per batch row.  S_in is [B, Lt] (target tokens so far, starting with BOS), E is [B, K, Le] and R is [B, K].  Returns
    # tok, pos, role [B, W] and tgt_idx [B, Lt], the column that holds each target token.  Exemplar tokens are compacted to the left with padding removed (positions restart at 0 for
    # each exemplar), the target segment follows directly, and anything after it is padding that no real token can attend to (causal mask), so no padding mask is needed.
    def pack(self, S_in, E, R):
        B, K, Le = E.shape
        Lt = S_in.size(1)
        dev = E.device
        flat = E.reshape(B, K * Le)
        keep = flat != 0
        n_ex = keep.sum(dim=1)
        n_max = int(n_ex.max().item()) if B > 0 else 0
        order = torch.argsort((~keep).to(torch.int32), dim=1, stable=True)
        ex_tok = torch.gather(flat, 1, order)[:, :n_max]
        pos_all = torch.arange(Le, device=dev).repeat(K).unsqueeze(0).expand(B, -1)
        ex_pos = torch.gather(pos_all, 1, order)[:, :n_max]
        role_all = R.unsqueeze(-1).expand(B, K, Le).reshape(B, K * Le)
        ex_role = torch.gather(role_all, 1, order)[:, :n_max]
        W = n_max + Lt
        tok = torch.zeros(B, W, dtype=torch.long, device=dev)
        pos = torch.zeros(B, W, dtype=torch.long, device=dev)
        role = torch.zeros(B, W, dtype=torch.long, device=dev)
        tok[:, :n_max] = ex_tok
        pos[:, :n_max] = ex_pos
        role[:, :n_max] = ex_role
        tgt_idx = n_ex.unsqueeze(1) + torch.arange(Lt, device=dev).unsqueeze(0)
        tok.scatter_(1, tgt_idx, S_in)
        pos.scatter_(1, tgt_idx, torch.arange(Lt, device=dev).unsqueeze(0).expand(B, -1).contiguous())
        role.scatter_(1, tgt_idx, torch.full((B, Lt), self.TARGET_ROLE, dtype=torch.long, device=dev))
        return tok, pos, role, tgt_idx

    # Scores every target position.  Packs the sequence, embeds tokens + roles + per-molecule positions, runs the causal blocks and reads the outputs at the target columns.  Without the
    # pointer channel the result is logits; with it the result is LOG-PROBABILITIES (a mixture of the vocabulary and copy distributions), which is also valid input to cross-entropy
    # and to temperature sampling.
    def target_logits(self, S_in, E, R):
        tok, pos, role, tgt_idx = self.pack(S_in, E, R)
        W = tok.size(1)
        x = self.smiles_embed(tok) * math.sqrt(self.d_model)
        x = x + self.role_embed(role)
        # positions restart at 0 for every molecule
        x = x + self.pos_encoder.pe[0, :, :][pos].to(x.dtype)
        x = self.pos_encoder.dropout(x)
        # causal mask over the whole packed sequence (True = may not attend)
        mask = torch.triu(torch.ones(W, W, dtype=torch.bool, device=tok.device), diagonal=1)
        for blk in self.blocks:
            x = blk(x, src_mask=mask)
        h = torch.gather(x, 1, tgt_idx.unsqueeze(-1).expand(-1, -1, self.d_model))
        logits = self.output_proj(h)
        if not self.pointer:
            return logits
        # pointer-generator: mix the vocabulary distribution with a copy distribution over the exemplar tokens
        B, Lt = S_in.shape
        n_max = W - Lt
        p_vocab = torch.softmax(logits.float(), dim=-1)
        # no exemplars anywhere in this batch: nothing to copy from
        if n_max == 0:
            return torch.log(p_vocab + 1e-9)
        n_ex = tgt_idx[:, 0]
        tok_ex, pos_ex = tok[:, :n_max], pos[:, :n_max]
        nxt_tok = torch.cat([tok_ex[:, 1:], torch.zeros(B, 1, dtype=torch.long, device=tok.device)], dim=1)
        nxt_pos = torch.cat([pos_ex[:, 1:], torch.full((B, 1), -1, dtype=torch.long, device=tok.device)], dim=1)
        col = torch.arange(n_max, device=tok.device).unsqueeze(0)
        # a column can be copied FROM if the next column is a real token of the same exemplar (that next token is what would be copied)
        valid = (col + 1 < n_ex.unsqueeze(1)) & (nxt_pos == pos_ex + 1)
        # attention from each target position over the exemplar columns decides which successor token to copy; g gates between copying and the vocabulary distribution (0 when there is nothing to copy)
        q = self.ptr_q(h)
        k = self.ptr_k(x[:, :n_max])
        scores = torch.einsum('bld,bcd->blc', q.float(), k.float()) / math.sqrt(self.d_model)
        scores = scores.masked_fill(~valid.unsqueeze(1), -1e4)
        attn = torch.softmax(scores, dim=-1) * valid.unsqueeze(1).float()
        p_copy = torch.zeros(B, Lt, self.vocab_size, dtype=torch.float32, device=tok.device)
        p_copy.scatter_add_(2, nxt_tok.unsqueeze(1).expand(-1, Lt, -1), attn)
        has = valid.any(dim=1).float().view(B, 1, 1)
        g = torch.sigmoid(self.ptr_gate(h).float()) * has
        p = g * p_copy + (1.0 - g) * p_vocab
        return torch.log(p + 1e-9)


# Builds a DrugDesignModel from a hyperparameter dict, ignoring keys its constructor does not take.
def build_base(hp):
    cls = DrugDesignModel
    allowed = set(inspect.signature(cls.__init__).parameters) - {'self'}
    kw = {k: v for k, v in hp.items() if k in allowed}
    return cls(**kw)


# Loads a pretrained base checkpoint.  Returns (base model with its weights, preprocessor rebuilt from the saved vocabularies, raw checkpoint dict).
def load_base_checkpoint(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    pp = DrugDataPreprocessor.from_saved_vocabularies(ckpt['preprocessor_data'])
    base = build_base(ckpt['model_hyperparams'])
    base.load_state_dict(ckpt['model_state_dict'])
    return base, pp, ckpt


# Loads an exemplar checkpoint of either format.  Returns (model in eval mode on the device, preprocessor, raw checkpoint).  The base model is rebuilt from the saved
# hyperparameters and wrapped, then the exemplar weights are loaded.  Exits with a message if the file is not an exemplar checkpoint.
def load_exemplar_checkpoint(path, device='cpu'):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if ckpt.get('format') not in (FORMAT, FORMAT_LM):
        raise SystemExit(f"{path} is not an exemplar checkpoint (format={ckpt.get('format')!r}).")
    pp = DrugDataPreprocessor.from_saved_vocabularies(ckpt['preprocessor_data'])
    base = build_base(ckpt['model_hyperparams'])
    if ckpt['format'] == FORMAT_LM:
        model = ExemplarLM(base, max_exemplars=ckpt['max_exemplars'], extra_layers=ckpt.get('extra_layers', 0),
                           pointer=ckpt.get('pointer', False))
    else:
        model = ExemplarModel(base, max_exemplars=ckpt['max_exemplars'])
    model.load_state_dict(ckpt['exemplar_state_dict'])
    model.to(device).eval()
    return model, pp, ckpt


# Returns a function that encodes a SMILES string to its integer array, caching the results.
def make_arr_fn(pp):
    cache = {}

    def arr(key):
        a = cache.get(key)
        if a is None:
            a = pp.encode_smiles(key)
            cache[key] = a
        return a
    return arr


# Turns a batch of items (molecule, exemplar keys, roles) into numpy arrays S [B, Ls], E [B, k_max, Le] and R [B, k_max], trimmed to the longest molecule and exemplar in
# the batch.  At most k_max exemplars per item are used, each cut at ex_max_len tokens; unused slots and padding are 0.
def build_arrays(items, k_max, ex_max_len, arr_fn, ex_arr_fn=None, pad=0):
    ex_arr_fn = ex_arr_fn or arr_fn
    B = len(items)
    seqs = [arr_fn(k) for k, _, _ in items]
    Ls = max(2, max(int((s != pad).sum()) for s in seqs))
    S = np.stack([s[:Ls] for s in seqs]).astype(np.int64)
    exs = [[ex_arr_fn(e)[:ex_max_len] for e in ex[:k_max]] for _, ex, _ in items]
    Le = max([int((a != pad).sum()) for row in exs for a in row] + [1])
    E = np.zeros((B, k_max, Le), dtype=np.int64)
    R = np.zeros((B, k_max), dtype=np.int64)
    for b, (row, (_, _, roles)) in enumerate(zip(exs, items)):
        for j, a in enumerate(row):
            n = min(int((a != pad).sum()), Le)
            E[b, j, :n] = a[:n]
            R[b, j] = roles[j] if roles else 0
    return S, E, R


# Converts numpy arrays to long tensors on the device.
def to_device(arrays, device):
    return [torch.tensor(a, dtype=torch.long, device=device) for a in arrays]


# Float attention mask with -inf above the diagonal, so a position cannot see later positions.
def causal_mask(L, device):
    return torch.triu(torch.ones(L, L, device=device) * float('-inf'), diagonal=1)


# Teacher-forced cross-entropy per token: the model reads S[:, :-1] and is scored on S[:, 1:], padding ignored.  Returns (nll [B, L-1], mask [B, L-1] float); with amp=True on
# CUDA the forward pass runs in bfloat16.
def per_token_nll(model, S, E, R, pad=0, amp=False):
    di, dt = S[:, :-1], S[:, 1:]
    if amp and S.is_cuda:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model.target_logits(di, E, R)
    else:
        logits = model.target_logits(di, E, R)
    logits = logits.float()
    nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), dt.reshape(-1),
                                            ignore_index=pad, reduction='none').view(di.size(0), -1)
    return nll, (dt != pad).float()


# Average negative log-likelihood per token of each item's molecule under the model (no gradient, in chunks).  Returns (mean nll per item [N], token count per item [N]).
@torch.no_grad()
def sequence_nll(model, items, k_max, ex_max_len, arr_fn, pad=0, chunk=32, amp=False):
    model.eval()
    dev = next(model.parameters()).device
    out_nll, out_tok = [], []
    for i in range(0, len(items), chunk):
        S, E, R = to_device(build_arrays(items[i:i + chunk], k_max, ex_max_len, arr_fn, pad=pad), dev)
        nll, m = per_token_nll(model, S, E, R, pad=pad, amp=amp)
        out_nll.extend((nll * m).sum(1).tolist())
        out_tok.extend(m.sum(1).tolist())
    tok = np.maximum(np.array(out_tok), 1.0)
    return np.array(out_nll) / tok, tok


# Samples SMILES token by token for a batch of exemplar sets (E, R), starting from BOS, with a temperature-scaled softmax.  PAD and BOS can never be sampled, and a sequence that
# has produced EOS is padded until all sequences are done.  Returns the token tensor [n, L], BOS included.
@torch.no_grad()
def sample_tokens(model, pp, E, R, temperature=1.0, max_len=None):
    model.eval()
    dev = E.device
    BOS, EOS, PAD = pp.smiles_vocab['<BOS>'], pp.smiles_vocab['<EOS>'], pp.PAD_IDX
    max_len = max_len or pp.max_smiles_len
    n = E.size(0)
    tokens = torch.full((n, 1), BOS, dtype=torch.long, device=dev)
    done = torch.zeros(n, dtype=torch.bool, device=dev)
    for _ in range(max_len - 1):
        # the whole sequence is recomputed at every step (no key/value cache): slower, but identical to training
        if dev.type == 'cuda':
            # bfloat16 halves the memory of the attention matrices
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lg = model.target_logits(tokens, E, R)
        else:
            lg = model.target_logits(tokens, E, R)
        logits = lg[:, -1, :].float() / max(temperature, 1e-6)
        logits[:, PAD] = float('-inf')
        logits[:, BOS] = float('-inf')
        nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(1)
        nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
        tokens = torch.cat([tokens, nxt[:, None]], dim=1)
        done = done | (nxt == EOS)
        if bool(done.all()):
            break
    return tokens


# Turns sampled token rows into SMILES strings: skips the leading BOS and stops at the first EOS or PAD.
def decode_tokens(tokens, pp):
    EOS, PAD = pp.smiles_vocab['<EOS>'], pp.PAD_IDX
    inv = pp.smiles_inv_vocab
    res = []
    for row in tokens.tolist():
        chars = []
        for t in row[1:]:
            if t in (EOS, PAD):
                break
            chars.append(inv.get(t, ''))
        res.append(''.join(chars))
    return res


# Generates one SMILES per exemplar set (exemplar_sets is a list of (exemplar keys, roles)), in chunks.  Returns a list of strings, some of which may be empty or invalid SMILES.
def generate(model, pp, exemplar_sets, k_max, ex_max_len, arr_fn, temperature=1.0, chunk=24):
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(exemplar_sets), chunk):
        # the molecule slot is not used for sampling; 'C' is only a placeholder so that build_arrays can run
        items = [('C', ex, roles) for ex, roles in exemplar_sets[i:i + chunk]]
        _, E, R = build_arrays(items, k_max, ex_max_len, arr_fn, arr_fn, pad=pp.PAD_IDX)
        E, R = to_device([E, R], dev)
        out.extend(decode_tokens(sample_tokens(model, pp, E, R, temperature), pp))
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
    return out


# Lazy RDKit handle with caches: Morgan fingerprints (radius 2, 2048 bits) and canonical SMILES per string, plus the similarity helpers used by the evaluation scripts.
class Chem2:

    # Imports RDKit here (so it is only needed when this class is used), silences its logging and creates the fingerprint generator and the caches.
    def __init__(self):
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import rdFingerprintGenerator
        RDLogger.DisableLog('rdApp.*')
        self.Chem, self.DataStructs = Chem, DataStructs
        self.gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        self.cache = {}
        self.canon_cache = {}

    # Parses a SMILES; None for an empty or invalid string.
    def mol(self, smi):
        return self.Chem.MolFromSmiles(smi) if smi else None

    # Cached Morgan fingerprint of a SMILES (None if it does not parse).
    def fp(self, smi):
        if smi not in self.cache:
            m = self.mol(smi)
            self.cache[smi] = self.gen.GetFingerprint(m) if m is not None else None
        return self.cache[smi]

    # Fingerprints of a list of SMILES, skipping those that do not parse.
    def fps(self, smis):
        return [f for f in (self.fp(s) for s in smis) if f is not None]

    # Cached canonical SMILES (None if the string does not parse).
    def canon(self, smi):
        if smi not in self.canon_cache:
            m = self.mol(smi)
            self.canon_cache[smi] = self.Chem.MolToSmiles(m) if m is not None else None
        return self.canon_cache[smi]

    # Mean, over the fingerprints in q, of the best Tanimoto similarity to any fingerprint in pool; nan if either is empty.
    def mean_best(self, q, pool):
        if not q or not pool:
            return float('nan')
        return float(np.mean([max(self.DataStructs.BulkTanimotoSimilarity(f, pool)) for f in q]))

    # Tanimoto similarity of two SMILES (0.0 if either does not parse).
    def sim(self, a, b):
        fa, fb = self.fp(a), self.fp(b)
        return self.DataStructs.TanimotoSimilarity(fa, fb) if (fa is not None and fb is not None) else 0.0


# Lift of a similarity matrix S, where S[i, j] is the similarity of what was generated FOR i to the held-out molecules OF j.  Each column is divided by its mean, so 1.00 means
# no better than what the other contexts produce.  Returns (mean diagonal lift, per-row lifts).
def column_lift(S):
    S = np.array(S, dtype=float)
    col = np.nanmean(S, axis=0)
    lift = S / col[None, :]
    d = np.diag(lift)
    return float(np.nanmean(d)), d.tolist()


# Free-running test on held-out WHOLE targets (never seen in training).  Target contrast: for each held target, generate n molecules from random k-subsets of its lib and compare
# them with the TEST molecules of every held target (column-normalised lift, mean of the diagonal).  Disease contrast: at held targets with at least 2 populated diseases,
# generate from lib_by_d[d1] and lib_by_d[d2] and compare with test_by_d[d1] and test_by_d[d2] (own-disease similarity over other-disease similarity).  Reference: the same two
# numbers when the 'generated' molecules are random lib molecules (pure retrieval).  Returns a dict with targets, valid, target_lift, ref_target_lift, disease_lift,
# ref_disease_lift, dis_cases and, with keep=True, the samples.
def heldout_gen_eval(model, pp, sets, rng, k_max, ex_max_len, arr_fn, chem, k=5, n=20, max_targets=8,
                     temperature=1.0, max_disease_cases=6, keep=False):
    targets = [t for t in sets if len(sets[t]['lib']) >= 3][:max_targets]
    test_fps = {t: chem.fps(sets[t]['test'][:150]) for t in targets}
    targets = [t for t in targets if len(test_fps[t]) >= 3]
    res = {'targets': targets, 'samples': {}, 'dis_cases': []}
    if not targets:
        return res

    # Generates n molecules from random k-subsets of the pool, all with the same role.  Returns (exemplar sets, SMILES).
    def gen_from(pool, role):
        exs = []
        for _ in range(n):
            ex = rng.sample(pool, min(k, len(pool)))
            exs.append((ex, [role] * len(ex)))
        return exs, generate(model, pp, exs, k_max, ex_max_len, arr_fn, temperature)

    # S[i, j]: similarity of what was generated for target i to the test molecules of target j; S_ref is the same with random lib molecules in place of generated ones
    S = np.full((len(targets), len(targets)), np.nan)
    S_ref = np.full((len(targets), len(targets)), np.nan)
    valid = []
    for i, t in enumerate(targets):
        exs, smis = gen_from(sets[t]['lib'], xd.ROLE_TARGET)
        fps = chem.fps(smis)
        valid.append(len(fps) / max(len(smis), 1))
        ref = chem.fps(rng.sample(sets[t]['lib'], min(n, len(sets[t]['lib']))))
        for j, u in enumerate(targets):
            S[i, j] = chem.mean_best(fps, test_fps[u])
            S_ref[i, j] = chem.mean_best(ref, test_fps[u])
        if keep:
            res['samples'][t] = {'smiles': smis, 'exemplars': [e for e, _ in exs]}
    res['valid'] = float(np.mean(valid))
    res['valid_each'] = valid
    res['S'] = S
    res['target_lift'], res['target_lift_each'] = column_lift(S)
    res['ref_target_lift'], _ = column_lift(S_ref)

    # disease contrast: at targets with at least 2 populated diseases, compare the two with the most lib molecules
    cases = []
    for t in targets:
        ds = sorted(sets[t]['lib_by_d'], key=lambda d: -len(sets[t]['lib_by_d'][d]))
        if len(ds) >= 2:
            cases.append((t, ds[0], ds[1]))
    dl, dl_ref = [], []
    for t, d1, d2 in cases[:max_disease_cases]:
        s = sets[t]
        _, sm1 = gen_from(s['lib_by_d'][d1], xd.ROLE_PAIR)
        _, sm2 = gen_from(s['lib_by_d'][d2], xd.ROLE_PAIR)
        f1, f2 = chem.fps(sm1), chem.fps(sm2)
        p1, p2 = chem.fps(s['test_by_d'][d1][:150]), chem.fps(s['test_by_d'][d2][:150])
        if len(f1) < 5 or len(f2) < 5 or len(p1) < 3 or len(p2) < 3:
            continue
        # own-disease similarity over other-disease similarity, for the generated molecules and for the random-lib reference
        own = (chem.mean_best(f1, p1) + chem.mean_best(f2, p2)) / 2
        oth = (chem.mean_best(f1, p2) + chem.mean_best(f2, p1)) / 2
        r1 = chem.fps(rng.sample(s['lib_by_d'][d1], min(n, len(s['lib_by_d'][d1]))))
        r2 = chem.fps(rng.sample(s['lib_by_d'][d2], min(n, len(s['lib_by_d'][d2]))))
        own_r = (chem.mean_best(r1, p1) + chem.mean_best(r2, p2)) / 2
        oth_r = (chem.mean_best(r1, p2) + chem.mean_best(r2, p1)) / 2
        lift, lift_r = own / max(oth, 1e-9), own_r / max(oth_r, 1e-9)
        dl.append(lift)
        dl_ref.append(lift_r)
        res['dis_cases'].append({'target': t, 'd1': d1, 'd2': d2, 'lift': lift, 'ref_lift': lift_r,
                                 'n1': len(s['lib_by_d'][d1]), 'n2': len(s['lib_by_d'][d2])})
    res['disease_lift'] = float(np.mean(dl)) if dl else float('nan')
    res['ref_disease_lift'] = float(np.mean(dl_ref)) if dl_ref else float('nan')
    return res


# Reads the ligand CSV into rows [{key: SMILES string, dis: [str], tgt: [str]}] for every row with a usable SMILES (present, and short enough not to be truncated).  Diseases and
# targets stay as the raw strings of the CSV: the exemplar model needs no vocabulary.
def load_rows(csv_path, pp):
    import pandas as pd
    df = pd.read_csv(csv_path)
    P = pp._parse_list_field
    dis_l = df['Disease Entry'].apply(P)
    tgt_l = df['Targets for Drug'].apply(P)
    smi = df['Canonical Smiles']
    rows, n_skip = [], 0
    for i in df.index:
        key = smi.at[i]
        # rows whose SMILES is missing or too long to be encoded without truncation are skipped
        if not isinstance(key, str) or not key.strip() or len(key) + 2 > pp.max_smiles_len:
            n_skip += 1
            continue
        rows.append({'key': key,
                     'dis': [str(x).strip() for x in dis_l.at[i] if str(x).strip()],
                     'tgt': [str(x).strip() for x in tgt_l.at[i] if str(x).strip()]})
    print(f"[data] {len(df)} CSV rows; {n_skip} without a usable SMILES; {len(rows)} usable")
    return rows
