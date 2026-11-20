"""
base_model.py -- the base transformer (DrugDesignModel), the incremental (key/value cached) decoding helpers it uses, and the positional encoding.

The exemplar generator (exemplar_model.ExemplarModel / ExemplarLM) wraps a DrugDesignModel rebuilt from a checkpoint's hyper-parameters.
"""
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer


# Makes sure no row of a padding mask hides every position: attention over an all-masked row gives NaN, so position 0 is unmasked in any such row.  Returns the mask (a copy
# if it had to change).
def _guard_fully_masked_rows(padding_mask: torch.Tensor) -> torch.Tensor:
    fully_masked = padding_mask.all(dim=1)
    if fully_masked.any():
        padding_mask = padding_mask.clone()
        padding_mask[fully_masked, 0] = False
    return padding_mask


# Converts a padding mask in the nn.Transformer convention ([batch, mem_len], True = padding) to the one F.scaled_dot_product_attention takes (True = attend), shaped
# [batch, 1, 1, mem_len] so it broadcasts over the heads and the single query position of an incremental step.  None stays None.
def padding_mask_to_attend_mask(memory_key_padding_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if memory_key_padding_mask is None:
        return None
    return (~memory_key_padding_mask)[:, None, None, :]


# Reshapes [batch, seq, embed_dim] into [batch, nhead, seq, head_dim].
def _split_heads(x: torch.Tensor, nhead: int) -> torch.Tensor:
    batch, seq, embed_dim = x.shape
    head_dim = embed_dim // nhead
    return x.view(batch, seq, nhead, head_dim).transpose(1, 2)


# Reshapes [batch, nhead, seq, head_dim] back into [batch, seq, embed_dim].
def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    batch, nhead, seq, head_dim = x.shape
    return x.transpose(1, 2).contiguous().view(batch, seq, nhead * head_dim)


# Projects the encoder memory into one decoder layer's cross-attention keys and values, split into heads.  The memory does not change during a generation run, so this
# is done once per layer (from init_decoder_cache) and the result is reused at every step.
def compute_cross_attention_kv(layer: nn.Module, memory: torch.Tensor, nhead: int):
    w_q, w_k, w_v = layer.multihead_attn.in_proj_weight.chunk(3, dim=0)
    b = layer.multihead_attn.in_proj_bias
    if b is not None:
        _, b_k, b_v = b.chunk(3, dim=0)
    else:
        b_k = b_v = None

    k = F.linear(memory, w_k, b_k)
    v = F.linear(memory, w_v, b_v)

    return _split_heads(k, nhead), _split_heads(v, nhead)


# Self-attention of ONE new token (x_new, [batch, 1, embed_dim]) for one decoder layer.  The layer's cache of past keys and values is extended in place.  Only past
# tokens are in the cache, so the step is causal without a mask.
def _incremental_self_attention(layer: nn.Module, x_new: torch.Tensor, cache: dict, nhead: int) -> torch.Tensor:
    w_q, w_k, w_v = layer.self_attn.in_proj_weight.chunk(3, dim=0)
    b = layer.self_attn.in_proj_bias
    if b is not None:
        b_q, b_k, b_v = b.chunk(3, dim=0)
    else:
        b_q = b_k = b_v = None

    q = _split_heads(F.linear(x_new, w_q, b_q), nhead)
    k_new = _split_heads(F.linear(x_new, w_k, b_k), nhead)
    v_new = _split_heads(F.linear(x_new, w_v, b_v), nhead)

    if cache['self_k'] is None:
        k_full, v_full = k_new, v_new
    else:
        k_full = torch.cat([cache['self_k'], k_new], dim=2)
        v_full = torch.cat([cache['self_v'], v_new], dim=2)

    cache['self_k'], cache['self_v'] = k_full, v_full

    attn_dropout_p = layer.self_attn.dropout if layer.self_attn.training else 0.0
    attn_out = F.scaled_dot_product_attention(q, k_full, v_full, dropout_p=attn_dropout_p)
    attn_out = _merge_heads(attn_out)

    return layer.self_attn.out_proj(attn_out)


# Cross-attention of the new token against the layer's precomputed memory keys and values.  memory_attend_mask (True = attend) keeps the padded context slots out of
# the attention; without it the decoder would attend to padding as if it were content.
def _incremental_cross_attention(layer: nn.Module, x_new: torch.Tensor, cross_k: torch.Tensor, cross_v: torch.Tensor, nhead: int,
                                 memory_attend_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    w_q = layer.multihead_attn.in_proj_weight.chunk(3, dim=0)[0]
    b = layer.multihead_attn.in_proj_bias
    b_q = b.chunk(3, dim=0)[0] if b is not None else None

    q = _split_heads(F.linear(x_new, w_q, b_q), nhead)

    attn_dropout_p = layer.multihead_attn.dropout if layer.multihead_attn.training else 0.0
    attn_out = F.scaled_dot_product_attention(
        q, cross_k, cross_v, attn_mask=memory_attend_mask, dropout_p=attn_dropout_p
    )
    attn_out = _merge_heads(attn_out)

    return layer.multihead_attn.out_proj(attn_out)


# One decoder layer for one new token, in the order nn.TransformerDecoderLayer uses with norm_first=False (post-norm): self-attention, cross-attention, feed-forward, each
# followed by a residual add and a layer norm.  It reuses the layer's own weights, norms and dropout modules.
def _decoder_layer_forward_incremental(layer: nn.Module, x_new: torch.Tensor, self_cache: dict, cross_k: torch.Tensor, cross_v: torch.Tensor, nhead: int,
                                       memory_attend_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    sa_out = _incremental_self_attention(layer, x_new, self_cache, nhead)
    x = layer.norm1(x_new + layer.dropout1(sa_out))

    ca_out = _incremental_cross_attention(layer, x, cross_k, cross_v, nhead, memory_attend_mask)
    x = layer.norm2(x + layer.dropout2(ca_out))

    ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
    x = layer.norm3(x + layer.dropout3(ff_out))

    return x


# Runs all decoder layers for one new token, giving each layer its own cache and cross-attention keys and values.  cache_list is updated in place and the same mask is
# used in every layer.
def decoder_forward_incremental(decoder: nn.Module, x_new: torch.Tensor, cache_list: list, cross_kv_list: list, nhead: int,
                                memory_attend_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = x_new
    for i, layer in enumerate(decoder.layers):
        cross_k, cross_v = cross_kv_list[i]
        x = _decoder_layer_forward_incremental(layer, x, cache_list[i], cross_k, cross_v, nhead, memory_attend_mask)
    return x


# The base transformer: a SMILES token embedding, optional embeddings for the context fields (disease, target, KEGG, Wiki) with a context encoder, a decoder that attends to
# the encoded context, and an output projection onto the SMILES vocabulary.  Pretraining and the ID-conditioned model use all of it; the exemplar generator builds on its
# pretrained weights.
class DrugDesignModel(nn.Module):
    # Builds the layers.  context_vocab_sizes maps each context field name to its vocabulary size (no context embeddings if it is omitted).  The attention layers are
    # batch-first and post-norm, the torch defaults.
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 context_vocab_sizes: dict = None):
        super(DrugDesignModel, self).__init__()

        self.d_model = embed_dim
        self.nhead = nhead
        self.vocab_size = vocab_size

        self.smiles_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # one embedding table per context field, so each field keeps its own vocabulary
        self.context_embeds = nn.ModuleDict()
        if context_vocab_sizes:
            for ctx_name, ctx_vocab_size in context_vocab_sizes.items():
                self.context_embeds[ctx_name] = nn.Embedding(ctx_vocab_size, embed_dim, padding_idx=0)

        self.pos_encoder = PositionalEncoding(embed_dim, dropout)

        # context encoder: reads the concatenated context embeddings
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # decoder: generates the SMILES and attends to the encoded context
        decoder_layer = TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.output_proj = nn.Linear(embed_dim, vocab_size)

        self.init_weights()

    # Small uniform initialisation (+-0.1) of the SMILES embedding and of the output projection; the transformer layers keep torch's default initialisation.
    def init_weights(self):
        initrange = 0.1
        self.smiles_embed.weight.data.uniform_(-initrange, initrange)
        self.output_proj.bias.data.zero_()
        self.output_proj.weight.data.uniform_(-initrange, initrange)

    # Encodes the context fields ({'disease', 'target', 'kegg', 'wiki'}: integer tensors, 0 = padding) into encoder memory: embed each field, concatenate, add positions and
    # run the encoder with the padding mask.  Returns (memory, mask), where mask is True at padding and has been guarded against all-masked rows; pass it on to the decoder as
    # it is.  Returns (None, None) if no field is recognised.  Every path that needs the memory (training, sampling, generation) goes through this method.
    def encode_context(self, context_dict: Dict[str, torch.Tensor]):
        context_emb_list = []
        context_pad_mask_list = []
        for ctx_name, ctx_tensor in context_dict.items():
            if ctx_name in self.context_embeds:
                context_emb_list.append(self.context_embeds[ctx_name](ctx_tensor))
                # 0 is the padding id in every vocabulary
                context_pad_mask_list.append(ctx_tensor == 0)

        if not context_emb_list:
            return None, None

        context_embedded = torch.cat(context_emb_list, dim=1)
        memory_key_padding_mask = _guard_fully_masked_rows(torch.cat(context_pad_mask_list, dim=1))

        context_embedded = self.pos_encoder(context_embedded)
        # padded positions (for example a field with 2 real entries in a 10-slot cap) are ignored by the encoder
        memory = self.encoder(context_embedded, src_key_padding_mask=memory_key_padding_mask)
        return memory, memory_key_padding_mask

    # Training forward pass: encodes the context, embeds the target SMILES tokens, runs the decoder (tgt_mask is the causal mask supplied by the caller) with cross-attention
    # to the context memory, and returns logits [batch, length, vocab].  src_smiles is used only when no context field is given; src_mask is not used.
    def forward(self,
                src_smiles: torch.Tensor,
                tgt_smiles: torch.Tensor,
                context_dict: Dict[str, torch.Tensor],
                src_mask=None,
                tgt_mask=None) -> torch.Tensor:
        memory, memory_key_padding_mask = self.encode_context(context_dict)

        # no context given: the source SMILES themselves are encoded and used as the memory (unmasked)
        if memory is None:
            src_embedded = self.smiles_embed(src_smiles) * math.sqrt(self.d_model)
            src_embedded = self.pos_encoder(src_embedded)
            memory = self.encoder(src_embedded)
            memory_key_padding_mask = None

        tgt_embedded = self.smiles_embed(tgt_smiles) * math.sqrt(self.d_model)
        tgt_embedded = self.pos_encoder(tgt_embedded)

        # the padding mask the encoder used also keeps the decoder's cross-attention off the padded context slots
        output = self.decoder(
            tgt_embedded, memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        output = self.output_proj(output)

        return output

    # Prepares the caches for one generation run: an empty self-attention cache per decoder layer, each layer's cross-attention keys and values for the memory (computed once),
    # and the padding mask converted to the attend convention.  Call it once per run, not once per step.  Returns (self_cache, cross_kv, cross_attn_mask); the mask is None if
    # none was given.  Without a memory cross_kv holds None per layer, which the incremental path does not support.
    def init_decoder_cache(self, memory: torch.Tensor, memory_key_padding_mask: Optional[torch.Tensor] = None):
        num_layers = len(self.decoder.layers)
        self_cache = [{'self_k': None, 'self_v': None} for _ in range(num_layers)]

        if memory is not None:
            cross_kv = [compute_cross_attention_kv(layer, memory, self.nhead) for layer in self.decoder.layers]
        else:
            cross_kv = [None] * num_layers

        cross_attn_mask = padding_mask_to_attend_mask(memory_key_padding_mask)

        return self_cache, cross_kv, cross_attn_mask

    # Runs ONE new token through the decoder, using and extending the caches, and returns the logits [batch, vocab] for the token that follows it.  position is the
    # token's absolute index in the sequence (0 for BOS).  Only past tokens are in the cache, so the step is causal by construction.
    def generate_step_incremental(self,
                                  new_token: torch.Tensor,
                                  position: int,
                                  self_cache: list,
                                  cross_kv: list,
                                  cross_attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.smiles_embed(new_token) * math.sqrt(self.d_model)
        x = self.pos_encoder.forward_incremental(x, position)

        x = decoder_forward_incremental(self.decoder, x, self_cache, cross_kv, self.nhead, cross_attn_mask)

        logits = self.output_proj(x)
        return logits[:, -1, :]


# Fixed sinusoidal position encodings, precomputed for up to max_len positions, added to the embeddings and followed by dropout.
class PositionalEncoding(nn.Module):
    # Precomputes the sine / cosine table and stores it as a (non-trained) buffer.
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    # Adds the encodings of positions 0 .. seq_len-1 to x [batch, seq_len, d_model] and applies dropout.
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    # Same as forward for ONE token at an absolute position: adds just that row of the table.  x is [batch, 1, d_model].
    def forward_incremental(self, x, position: int):
        x = x + self.pe[:, position:position + 1, :]
        return self.dropout(x)
