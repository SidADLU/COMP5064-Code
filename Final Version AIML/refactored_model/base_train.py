"""
base_train.py -- pretraining of the base language model (next-character SMILES prediction given the context ids), the agent that samples from it, and checkpoint save / load.

Not needed to generate or dock: the exemplar scripts only read the pretrained checkpoint (checkpoint_pretrained_v11.pt by default).  Run pretraining by importing these
functions (the old script ran it at the top level; see legacy/ for that driver code).

The checkpoint path is built from config.py.
"""
import copy
import os
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from rdkit import Chem
from tqdm import tqdm

import config as cfg
from config import SMOKE_TEST
from base_data import DrugDataPreprocessor
from base_model import DrugDesignModel

# mixed precision: bfloat16 on a CUDA GPU that supports it, else float16 on CUDA, else plain float32 on the CPU
AMP_ENABLED = torch.cuda.is_available()
if AMP_ENABLED and torch.cuda.is_bf16_supported():
    AMP_DTYPE = torch.bfloat16
elif AMP_ENABLED:
    AMP_DTYPE = torch.float16
else:
    AMP_DTYPE = torch.float32

# torch.compile of the per-token decoder call (CUDA only); switched off
USE_TORCH_COMPILE = False
# written into every checkpoint: 2 = trained with the padding mask on the decoder's cross-attention.  A checkpoint with a lower number (or none) gets a warning when loaded
CONTEXT_MASKING_VERSION = 2
# file-name tag of the pretraining checkpoint; the smoke-test run gets its own file
CKPT_TAG = 'v11_smoke' if SMOKE_TEST else 'v11'
# default pretraining checkpoint: in the same folder as the pretrained checkpoint named in config.py
PRETRAIN_CHECKPOINT_PATH = os.path.join(os.path.dirname(cfg.DEFAULT_PRETRAINED_CKPT), 'checkpoint_pretrained_' + CKPT_TAG + '.pt')
# pretraining settings: epoch limit, early-stopping patience (epochs without a better validation loss) and the Adam learning rate
PRETRAIN_MAX_EPOCHS = 10
PRETRAIN_PATIENCE = 3
PRETRAIN_LR = 1e-3


# Returns the autocast context for this machine: bfloat16 / float16 autocast on CUDA, a disabled autocast on the CPU.  The weights stay float32; autocast only lowers the
# precision of the operations that support it.
def amp_context():
    if AMP_ENABLED:
        return torch.autocast(
            device_type="cuda",
            dtype=AMP_DTYPE
        )

    return torch.autocast(
        device_type="cpu",
        enabled=False
    )


# Wraps a DrugDesignModel for sampling: generates SMILES token by token with a key/value cache, decodes and checks them with RDKit, and returns the log-probabilities,
# entropy and KL values that a policy-gradient update needs.  A frozen reference model can be attached as ref_model for the KL term.
class DrugGeneratorAgent:
    # Moves the model to the device, optionally compiles the per-step decoder call, and takes the special-token ids and the maximum length from the preprocessor.
    # The sampling temperature starts at 1.0 and ref_model at None.
    def __init__(self, model: DrugDesignModel, preprocessor, device='cpu'):
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        self.model.to(device)

        # compile the per-step decoder call once, here (not per batch); dynamic=True because the cache grows by one position every step and each new length would otherwise trigger a recompile
        if USE_TORCH_COMPILE and device == 'cuda':
            self.model.generate_step_incremental = torch.compile(
                self.model.generate_step_incremental,
                dynamic=True,
            )


        self.inv_vocab = preprocessor.smiles_inv_vocab
        self.pad_idx = preprocessor.PAD_IDX
        self.bos_idx = preprocessor.BOS_IDX
        self.eos_idx = preprocessor.EOS_IDX

        self.temperature = 1.0
        self.max_len = preprocessor.max_smiles_len

        # frozen copy of the pretrained model used as the reference for the KL penalty; None until it is set before RL training
        self.ref_model: Optional[DrugDesignModel] = None

    # True if RDKit can parse the SMILES string (an empty string or a parse error counts as invalid).
    def _check_validity(self, smiles: str) -> bool:
        if not smiles:
            return False
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False

    # Turns token ids into a SMILES string: skips PAD and BOS, stops at the first EOS and writes '<UNK>' for an id that is not in the vocabulary.
    def _decode_tokens(self, tokens: List[int]) -> str:
        decoded_tokens = []
        for idx in tokens:
            if idx == self.pad_idx: continue
            if idx == self.bos_idx: continue
            if idx == self.eos_idx: break
            if idx in self.inv_vocab:
                decoded_tokens.append(self.inv_vocab[idx])
            else:
                decoded_tokens.append('<UNK>')
        return "".join(decoded_tokens)

    # Samples batch_size SMILES in parallel for the given context, one token per step with a key/value cache, and returns what a policy-gradient update needs: (sequences,
    # log_probs [batch, max_len], entropy_sum, kl_sum, seq_lengths).  Steps after a sequence's own EOS are masked out of every returned quantity (the EOS step itself counts), so
    # a finished sequence does not collect values for tokens that never appear in its SMILES.  entropy_sum is the summed per-step entropy; kl_sum is the summed log-ratio against
    # ref_model for the sampled tokens (zeros when no reference model is set).  The model is left in train mode, so dropout is active during sampling.
    def generate_batch_with_log_probs(self,
                                      context_dict: Dict[str, torch.Tensor],
                                      batch_size: int = 32,
                                      max_len: int = None) -> Tuple[List[List[int]], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if max_len is None:
            max_len = self.max_len

        self.model.train()
        if self.ref_model is not None:
            self.ref_model.eval()

        # the context memory and the decoder caches are computed once, before the sampling loop
        with amp_context():
            memory, memory_key_padding_mask = self.model.encode_context(context_dict)

            self_cache, cross_kv, cross_attn_mask = self.model.init_decoder_cache(
                memory, memory_key_padding_mask
            )

        current_tokens = torch.full(
            (batch_size, 1),
            self.bos_idx,
            device=self.device,
            dtype=torch.long
        )

        step_log_probs = []
        step_entropies = []
        step_masks = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        new_token = current_tokens

        # sampling loop: one new token per step, until every sequence has produced EOS or max_len is reached
        for step in range(max_len - 1):
            # a step counts for a sequence only if the sequence had not finished before the step began: the step that chooses EOS counts, later steps do not
            active_before_step = ~finished

            with amp_context():
                logits = self.model.generate_step_incremental(
                    new_token,
                    position=step,
                    self_cache=self_cache,
                    cross_kv=cross_kv,
                    cross_attn_mask=cross_attn_mask
                )

            # probabilities are computed in float32 because they feed the policy-gradient loss, even though the decoder runs in lower precision
            logits = logits.float()
            logits = logits / self.temperature

            log_probs = F.log_softmax(logits, dim=-1)

            probs = torch.exp(log_probs)

            # entropy of the whole distribution at this step, not just of the sampled token
            step_entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)
            step_entropies.append(step_entropy)

            next_token = torch.multinomial(probs, 1)

            chosen_log_prob = torch.gather(
                log_probs,
                dim=1,
                index=next_token
            )

            step_log_probs.append(chosen_log_prob)
            step_masks.append(active_before_step)

            current_tokens = torch.cat(
                (current_tokens, next_token),
                dim=1
            )
            new_token = next_token

            is_eos = (next_token.squeeze(-1) == self.eos_idx)
            finished = finished | is_eos

            if finished.all():
                break

        # gather the steps and zero everything after each sequence's own EOS
        total_log_probs = torch.cat(step_log_probs, dim=1)
        total_step_entropy = torch.cat(step_entropies, dim=1)
        mask = torch.stack(step_masks, dim=1).float()

        total_log_probs = total_log_probs * mask
        total_step_entropy = total_step_entropy * mask

        # number of counted steps per sequence (the clamp is only a guard: step 0 always counts)
        seq_lengths = mask.sum(dim=1).clamp(min=1.0)

        entropy_sum = total_step_entropy.sum(dim=1)

        # KL term: one teacher-forced pass of the frozen reference model over the tokens just sampled gives its log-probability for each of them
        num_steps_taken = total_log_probs.shape[1]
        if self.ref_model is not None:
            decoder_input = current_tokens[:, :-1]
            decoder_target = current_tokens[:, 1:]
            tgt_mask = torch.triu(
                torch.ones(num_steps_taken, num_steps_taken) * float('-inf'), diagonal=1
            ).to(self.device)

            with torch.no_grad(), amp_context():
                ref_logits = self.ref_model.forward(
                    decoder_input, decoder_input, context_dict, tgt_mask=tgt_mask
                )
            ref_logits = ref_logits.float() / self.temperature
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            ref_chosen_log_prob = torch.gather(
                ref_log_probs, dim=-1, index=decoder_target.unsqueeze(-1)
            ).squeeze(-1)

            # masked again on purpose: the reference log-probabilities are not zero after EOS, so without the mask the difference would include its score for tokens that never appear
            kl_sum = ((total_log_probs - ref_chosen_log_prob) * mask).sum(dim=1)
        else:
            kl_sum = torch.zeros(batch_size, device=self.device)

        sequences = current_tokens.tolist()

        # pad the log-probabilities with zeros up to max_len columns (the loop stops early when all sequences finish)
        if total_log_probs.shape[1] < max_len:
            padding = torch.zeros(batch_size, max_len - total_log_probs.shape[1], device=self.device)
            total_log_probs = torch.cat((total_log_probs, padding), dim=1)

        return sequences, total_log_probs, entropy_sum, kl_sum, seq_lengths

    # Samples one sequence at a time (eval mode, no gradient) until num_samples SMILES are valid, not in known_smiles_set (compared in canonical form) and not already produced, or
    # max_attempts is reached; prints a warning if fewer were found.  The context memory and the cross-attention keys and values are computed once; the self-attention cache is
    # rebuilt for each attempt.  Returns the accepted SMILES as sampled (not canonicalised).
    def generate_unique_valid_smiles(self,
                                     context_dict: Dict[str, torch.Tensor],
                                     known_smiles_set: Set[str] = None,
                                     num_samples: int = 100,
                                     max_attempts: int = 1000) -> List[str]:
        self.model.eval()
        generated_smiles_list = []
        seen_smiles = set()

        if known_smiles_set is None:
            known_smiles_set = set()
        else:
            known_smiles_set = set(known_smiles_set)

        attempts = 0

        with torch.no_grad():
            # context memory, computed once for the whole call
            memory, memory_key_padding_mask = self.model.encode_context(context_dict)

            # the memory's keys and values and the attend mask depend only on the context, so they are computed once; the self-attention cache is rebuilt for every attempt below
            _, cross_kv, cross_attn_mask = self.model.init_decoder_cache(memory, memory_key_padding_mask)

            while len(generated_smiles_list) < num_samples and attempts < max_attempts:
                attempts += 1

                batch_size = 1
                current_tokens = torch.full((batch_size, 1), self.bos_idx, device=self.device, dtype=torch.long)
                new_token = current_tokens

                self_cache = [{'self_k': None, 'self_v': None} for _ in range(len(self.model.decoder.layers))]

                # token-by-token sampling with temperature scaling, until EOS or the maximum length
                for position in range(self.max_len - 1):
                    logits = self.model.generate_step_incremental(
                        new_token, position=position, self_cache=self_cache, cross_kv=cross_kv,
                        cross_attn_mask=cross_attn_mask
                    )

                    logits = logits / self.temperature

                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)

                    current_tokens = torch.cat((current_tokens, next_token), dim=1)
                    new_token = next_token

                    if next_token.item() == self.eos_idx:
                        break

                tokens_list = current_tokens[0].tolist()
                smiles = self._decode_tokens(tokens_list)

                # an attempt is kept only if it is valid SMILES, is not a known molecule (canonical SMILES compared) and has not been produced already
                if not self._check_validity(smiles):
                    continue

                canonical_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
                if canonical_smiles in known_smiles_set:
                    continue

                if smiles in seen_smiles:
                    continue

                seen_smiles.add(smiles)
                generated_smiles_list.append(smiles)

                if len(generated_smiles_list) == num_samples:
                    break

        if len(generated_smiles_list) < num_samples:
            print(f"Warning: Only generated {len(generated_smiles_list)} unique valid novel SMILES out of {num_samples} requested.")

        return generated_smiles_list


# Builds a DrugDesignModel sized to the preprocessor's vocabularies (the SMILES vocabulary plus the four context fields).  The architecture is 256-dimensional with 8 heads,
# 3 encoder and 3 decoder layers and a 1024-wide feed-forward (a tiny version in SMOKE_TEST mode).  The settings are stored on model.hyperparams so that save_checkpoint can
# record them.  Returns the model.
def initialize_model(preprocessor):
    smiles_vocab_size = len(preprocessor.smiles_vocab)

    # vocabulary size of each context field
    context_vocab_sizes = {
        'disease': len(preprocessor.disease_vocab),
        'target': len(preprocessor.target_vocab),
        'kegg': len(preprocessor.kegg_vocab),
        'wiki': len(preprocessor.wiki_vocab)
    }

    # the architecture; saved in checkpoints so the model can be rebuilt identically
    model_hyperparams = {
        'vocab_size': len(preprocessor.smiles_vocab),
        'embed_dim': 32 if SMOKE_TEST else 256,
        'nhead': 2 if SMOKE_TEST else 8,
        'num_encoder_layers': 1 if SMOKE_TEST else 3,
        'num_decoder_layers': 1 if SMOKE_TEST else 3,
        'dim_feedforward': 64 if SMOKE_TEST else 1024,
        'dropout': 0.1,
        'context_vocab_sizes': context_vocab_sizes,
    }
    try:
        model = DrugDesignModel(**model_hyperparams)
    except NameError:
        raise NameError("DrugDesignModel class is not defined. Please ensure the model class is imported or defined before calling this function.")

    # remember how the model was built, for save_checkpoint
    model.hyperparams = model_hyperparams

    return model


# Saves everything needed to generate on another machine: the model weights, the architecture hyperparameters, the preprocessor's vocabularies and length limits, the
# temperature and optional extra_info (informational only).  The original CSV is not needed to load it.  The model must carry a .hyperparams attribute (initialize_model
# sets it), otherwise an AttributeError is raised.
def save_checkpoint(agent: 'DrugGeneratorAgent', filepath: str, extra_info: dict = None):
    if not hasattr(agent.model, 'hyperparams'):
        raise AttributeError(
            "agent.model has no 'hyperparams' attribute. Build the model via "
            "initialize_model(preprocessor) so its architecture is recorded, "
            "or set model.hyperparams manually before saving."
        )

    preprocessor = agent.preprocessor

    checkpoint = {
        # records that this model was trained with the cross-attention padding mask
        'context_masking_version': CONTEXT_MASKING_VERSION,
        'model_state_dict': agent.model.state_dict(),
        'model_hyperparams': agent.model.hyperparams,
        'agent_hyperparams': {
            'temperature': agent.temperature,
        },
        'preprocessor_data': {
            'max_smiles_len': preprocessor.max_smiles_len,
            'max_target_len': preprocessor.max_target_len,
            'max_kegg_len': preprocessor.max_kegg_len,
            'max_wiki_len': preprocessor.max_wiki_len,
            'max_disease_len': preprocessor.max_disease_len,
            'smiles_vocab': preprocessor.smiles_vocab,
            'disease_vocab': preprocessor.disease_vocab,
            'target_vocab': preprocessor.target_vocab,
            'kegg_vocab': preprocessor.kegg_vocab,
            'wiki_vocab': preprocessor.wiki_vocab,
            'PAD_IDX': preprocessor.PAD_IDX,
            'UNK_IDX': preprocessor.UNK_IDX,
            'BOS_IDX': preprocessor.BOS_IDX,
            'EOS_IDX': preprocessor.EOS_IDX,
        },
        'extra_info': extra_info or {},
    }

    torch.save(checkpoint, filepath)
    size_mb = os.path.getsize(filepath) / 1e6
    print(f"✅ Checkpoint saved to '{filepath}' ({size_mb:.1f} MB).")


# Rebuilds a DrugGeneratorAgent from a checkpoint written by save_checkpoint, on any machine and without the training data: the preprocessor from the saved vocabularies,
# the model from the saved hyperparameters, then the weights.  device defaults to CUDA if available, else the CPU.  The agent is returned in eval mode.  A checkpoint trained
# before the cross-attention padding mask existed still loads but prints a warning.
def load_trained_agent(checkpoint_path: str, device: str = None) -> 'DrugGeneratorAgent':
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # map_location moves the tensors to a device that exists on this machine; weights_only=False because the file also holds vocabularies and hyperparameters (only load checkpoints you trust)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # older checkpoints were trained without the padding mask that generation now uses, so they load with a warning
    if checkpoint.get('context_masking_version', 1) < CONTEXT_MASKING_VERSION:
        print(f"⚠️  '{checkpoint_path}' was trained BEFORE the v11 cross-attention padding-mask fix "
              f"(context_masking_version={checkpoint.get('context_masking_version', 1)} < {CONTEXT_MASKING_VERSION}). "
              f"It will still load, but v11 now masks padding at generation time and this model was "
              f"trained without that mask, so its outputs are not representative of a properly "
              f"conditioned model. Retrain (pretraining + RL) for a valid v11 result.")

    preprocessor = DrugDataPreprocessor.from_saved_vocabularies(checkpoint['preprocessor_data'])

    # rebuild the model with the architecture it was trained with, then load the weights
    hp = checkpoint['model_hyperparams']
    model = DrugDesignModel(
        vocab_size=hp['vocab_size'],
        embed_dim=hp['embed_dim'],
        nhead=hp['nhead'],
        num_encoder_layers=hp['num_encoder_layers'],
        num_decoder_layers=hp['num_decoder_layers'],
        dim_feedforward=hp['dim_feedforward'],
        dropout=hp['dropout'],
        context_vocab_sizes=hp['context_vocab_sizes'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.hyperparams = hp

    agent = DrugGeneratorAgent(model, preprocessor, device=device)
    agent.temperature = checkpoint['agent_hyperparams'].get('temperature', 1.0)

    agent.model.eval()

    print(f"✅ Loaded checkpoint from '{checkpoint_path}' onto device '{device}'.")
    if checkpoint.get('extra_info'):
        print(f"   Extra info: {checkpoint['extra_info']}")

    return agent


# Counts, for every disease index, how many dataset rows contain it (padding ignored): a measure of how well each disease is represented in pretraining.  Returns a Counter.
def compute_disease_train_frequency(dataset, pad_idx=0):
    freq = Counter()
    for i in range(len(dataset)):
        for idx in dataset[i]['disease'].tolist():
            if idx != pad_idx:
                freq[idx] += 1
    return freq


# Validation split by how well a row's diseases are represented in training: each row goes into 'rare', 'medium' or 'common' by the training frequency of its LEAST frequent
# disease (tertile1 and tertile2 are the frequency cut-offs), and the per-token loss and accuracy are printed for each bucket.  It shows whether pretraining is worse on rare
# diseases than the single validation loss reveals.  Rows with no disease are skipped.
@torch.no_grad()
def evaluate_by_disease_frequency(agent, val_loader, disease_freq, tertile1, tertile2, device):
    agent.model.eval()
    pad_idx = agent.pad_idx
    vocab_size = agent.model.vocab_size

    bucket_loss = {'rare': 0.0, 'medium': 0.0, 'common': 0.0}
    bucket_correct = {'rare': 0, 'medium': 0, 'common': 0}
    bucket_tokens = {'rare': 0, 'medium': 0, 'common': 0}

    # Maps a training frequency to its bucket name.
    def bucket_for_freq(f):
        if f <= tertile1:
            return 'rare'
        elif f <= tertile2:
            return 'medium'
        return 'common'

    for batch_data in val_loader:
        full_smiles = batch_data['smiles'].to(device)
        disease_ids = batch_data['context_dict']['disease']
        context_dict = {k: v.to(device) for k, v in batch_data['context_dict'].items()}

        # teacher forcing: the model reads tokens 0 .. n-2 and must predict tokens 1 .. n-1, with a causal mask
        decoder_input = full_smiles[:, :-1]
        decoder_target = full_smiles[:, 1:]
        seq_len = decoder_input.size(1)
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(device)

        logits = agent.model.forward(decoder_input, decoder_input, context_dict, tgt_mask=tgt_mask).float()
        per_token_loss = F.cross_entropy(
            logits.reshape(-1, vocab_size), decoder_target.reshape(-1),
            ignore_index=pad_idx, reduction='none'
        ).reshape(decoder_target.shape)

        real_token_mask = (decoder_target != pad_idx)
        correct_mask = (logits.argmax(dim=-1) == decoder_target) & real_token_mask

        for row in range(disease_ids.size(0)):
            row_diseases = [d for d in disease_ids[row].tolist() if d != pad_idx]
            if not row_diseases:
                continue
            worst_freq = min(disease_freq.get(d, 0) for d in row_diseases)
            bucket = bucket_for_freq(worst_freq)

            bucket_loss[bucket] += per_token_loss[row][real_token_mask[row]].sum().item()
            bucket_correct[bucket] += correct_mask[row].sum().item()
            bucket_tokens[bucket] += real_token_mask[row].sum().item()

    print("  Validation by disease representation:")
    for bucket in ['rare', 'medium', 'common']:
        tok = bucket_tokens[bucket]
        if tok > 0:
            print(f"    {bucket:>6}: loss={bucket_loss[bucket]/tok:.4f} "
                  f"acc={bucket_correct[bucket]/tok:.2%} ({tok} tokens)")
        else:
            print(f"    {bucket:>6}: no validation rows in this bucket")


# Free-running check of pretraining: samples a few molecules through the real generation path (no teacher forcing) for the context of the first validation row and prints how
# many are valid SMILES.  known_smiles_set is None on purpose: early in pretraining, reproducing a training molecule is the wanted signal, not something to filter out.
@torch.no_grad()
def check_pretrain_samples(agent, val_loader, device, num_samples=5):
    agent.model.eval()
    batch_data = next(iter(val_loader))
    context_dict = {k: v[:1].to(device) for k, v in batch_data['context_dict'].items()}

    smiles_list = agent.generate_unique_valid_smiles(
        context_dict, known_smiles_set=None, num_samples=num_samples, max_attempts=num_samples * 4
    )
    print(f"  Sample check: {len(smiles_list)}/{num_samples} valid molecules generated. Examples: {smiles_list[:3]}")


# Pretrains the base model on next-character prediction of the SMILES given the context: Adam, gradient clipping at 1.0 and, for float16 only, a loss scaler.  Each epoch
# trains, validates, prints a sample check and the validation split by disease frequency, and keeps the best-validation weights; training stops after `patience` epochs without
# improvement.  The best weights are restored and saved to checkpoint_path through save_checkpoint.
def pretrain_model(agent, train_loader, val_loader, device,
                    max_epochs=PRETRAIN_MAX_EPOCHS, patience=PRETRAIN_PATIENCE, lr=PRETRAIN_LR,
                    checkpoint_path=PRETRAIN_CHECKPOINT_PATH):
    pretrain_optimizer = torch.optim.Adam(agent.model.parameters(), lr=lr)
    pretrain_scaler = torch.amp.GradScaler('cuda', enabled=(AMP_DTYPE == torch.float16))

    pad_idx = agent.pad_idx
    vocab_size = agent.model.vocab_size

    # disease frequency tertiles, used to bucket validation rows by how well their diseases are represented
    disease_freq = compute_disease_train_frequency(train_loader.dataset, pad_idx=pad_idx)
    freq_values = sorted(disease_freq.values())
    tertile1 = freq_values[len(freq_values) // 3]
    tertile2 = freq_values[2 * len(freq_values) // 3]

    best_val_loss = float('inf')
    best_state_dict = None
    epochs_without_improvement = 0

    # Runs one pass over a loader, training when train is True and validating otherwise.  Returns (mean loss per real token, token accuracy).  The targets are the SMILES shifted
    # by one position; PAD positions are ignored.
    def run_epoch(loader, train: bool, desc: str):
        agent.model.train() if train else agent.model.eval()
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0

        pbar = tqdm(loader, desc=desc, unit="batch")
        for batch_data in pbar:
            full_smiles = batch_data['smiles'].to(device)
            context_dict = {k: v.to(device) for k, v in batch_data['context_dict'].items()}

            decoder_input = full_smiles[:, :-1]
            decoder_target = full_smiles[:, 1:]
            seq_len = decoder_input.size(1)

            # causal mask: a position cannot see later tokens
            tgt_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(device)

            with torch.set_grad_enabled(train):
                with amp_context():
                    logits = agent.model.forward(
                        decoder_input, decoder_input, context_dict, tgt_mask=tgt_mask
                    )
                logits = logits.float()

                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    decoder_target.reshape(-1),
                    ignore_index=pad_idx
                )

                # optimiser step with gradient clipping at 1.0 (the scaler only has an effect with float16)
                if train:
                    pretrain_optimizer.zero_grad()
                    pretrain_scaler.scale(loss).backward()
                    pretrain_scaler.unscale_(pretrain_optimizer)
                    torch.nn.utils.clip_grad_norm_(agent.model.parameters(), max_norm=1.0)
                    pretrain_scaler.step(pretrain_optimizer)
                    pretrain_scaler.update()

            real_token_mask = (decoder_target != pad_idx)
            num_real_tokens = real_token_mask.sum().item()
            num_correct = ((logits.argmax(dim=-1) == decoder_target) & real_token_mask).sum().item()

            total_loss += loss.item() * num_real_tokens
            total_tokens += num_real_tokens
            total_correct += num_correct
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / max(total_tokens, 1), total_correct / max(total_tokens, 1)

    # epoch loop with early stopping on the validation loss
    for epoch in range(max_epochs):
        train_loss, train_acc = run_epoch(train_loader, train=True, desc=f"Pretrain {epoch+1}/{max_epochs} [train]")
        val_loss, val_acc = run_epoch(val_loader, train=False, desc=f"Pretrain {epoch+1}/{max_epochs} [val]")

        print(f"Pretrain Epoch {epoch+1}/{max_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2%}")
        check_pretrain_samples(agent, val_loader, device)
        evaluate_by_disease_frequency(agent, val_loader, disease_freq, tertile1, tertile2, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(agent.model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"No val improvement for {patience} epochs -- stopping pretraining early.")
                break

    # restore the best-validation weights before saving
    agent.model.load_state_dict(best_state_dict)
    print(f"Pretraining complete. Best val loss: {best_val_loss:.4f}")

    save_checkpoint(agent, filepath=checkpoint_path,
                     extra_info={'phase': 'pretrain', 'best_val_loss': best_val_loss})
