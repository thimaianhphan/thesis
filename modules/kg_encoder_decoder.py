"""
R2Gen + Knowledge Graph + Contrastive Attention (v2+CA)

Data flow:
  Image → ResNet → fc_feats → KG Encoder (GCN + image gate) → kg_feats [B,N,D]
                 → att_feats → [CA] → Transformer Encoder → encoder_out
                                                                    ↓
  Decoder: self_attn(+MCLN) → visual_cross_attn(+MCLN) → KG_cross_attn(gated) → FFN(+MCLN)
                                                                    ↓
                                                              logits → report

KG fuses at DECODER level (KiUT-style), NOT encoder level.
No DCG bidirectional cross-attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math

from .visual_extractor import VisualExtractor
from .encoder_decoder import (
    Encoder, EncoderLayer, MultiHeadedAttention, PositionwiseFeedForward,
    PositionalEncoding, Embeddings, RelationalMemory,
    ConditionalSublayerConnection, LayerNorm, clones, subsequent_mask
)
from .att_model import pack_wrapper, AttModel
from .knowledge_graph import (
    KnowledgeGraphBuilder, KnowledgeGraphEncoder, KGCrossAttention,
    KGMultiLabelClassifier, KGAlignmentLoss
)
from .contrastive_attention import ContrastiveAttention


# =============================================================================
# KG Decoder Layer: self-attn → visual cross-attn → KG cross-attn → FFN
# =============================================================================

class KGDecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout,
                 rm_num_slots, rm_d_model, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(
            ConditionalSublayerConnection(d_model, dropout, rm_num_slots, rm_d_model), 3
        )
        # KG cross-attention uses 4 heads (not the main Transformer's 8)
        self.kg_cross_attn = KGCrossAttention(d_model, num_heads=4, dropout=dropout)

    def forward(self, x, hidden_states, src_mask, tgt_mask, memory, kg_feats=None):
        m = hidden_states
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask), memory)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask), memory)
        if kg_feats is not None:
            x = self.kg_cross_attn(x, kg_feats)
        return self.sublayer[2](x, self.feed_forward, memory)


class KGDecoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, hidden_states, src_mask, tgt_mask, memory, kg_feats=None):
        for layer in self.layers:
            x = layer(x, hidden_states, src_mask, tgt_mask, memory, kg_feats)
        return self.norm(x)


class KGTransformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, rm):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.rm = rm

    def forward(self, src, tgt, src_mask, tgt_mask, kg_feats=None):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask, kg_feats)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, hidden_states, src_mask, tgt, tgt_mask, kg_feats=None):
        memory = self.rm.init_memory(hidden_states.size(0)).to(hidden_states)
        memory = self.rm(self.tgt_embed(tgt), memory)
        return self.decoder(self.tgt_embed(tgt), hidden_states, src_mask, tgt_mask, memory, kg_feats)


# =============================================================================
# KG-Enhanced EncoderDecoder
# =============================================================================

class KGEncoderDecoder(AttModel):

    def make_model(self, tgt_vocab):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        rm = RelationalMemory(
            num_slots=self.rm_num_slots, d_model=self.rm_d_model,
            num_heads=self.rm_num_heads
        )
        model = KGTransformer(
            Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers),
            KGDecoder(
                KGDecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout,
                               self.rm_num_slots, self.rm_d_model, self.num_heads),
                self.num_layers
            ),
            lambda x: x,
            nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position)),
            rm
        )
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
        self.args = args
        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.rm_num_slots = args.rm_num_slots
        self.rm_num_heads = args.rm_num_heads
        self.rm_d_model = args.rm_d_model

        tgt_vocab = self.vocab_size + 1

        # Build KG
        kg_builder = KnowledgeGraphBuilder(
            args.ann_path, args.dataset_name,
            co_occur_threshold=getattr(args, 'kg_co_occur_threshold', 3),
        )
        node_list, node_types, adjacency, node2idx = kg_builder.build(split='train')
        self.node_list = node_list
        self.node_types = node_types
        self.node2idx = node2idx
        self.kg_builder = kg_builder
        self.num_kg_nodes = len(node_list)
        self.register_buffer('adj', torch.FloatTensor(adjacency))

        d_visual = args.d_vf * (2 if args.dataset_name == 'iu_xray' else 1)

        self.kg_encoder = KnowledgeGraphEncoder(
            num_nodes=self.num_kg_nodes, node_types=node_types,
            d_model=args.d_model, d_visual=d_visual,
            num_gcn_layers=getattr(args, 'kg_num_gcn_layers', 1),
            dropout=args.dropout,
            gcn_residual_alpha=getattr(args, 'kg_gcn_alpha', 0.2),
        )

        self.model = self.make_model(tgt_vocab)
        self.logit = nn.Linear(args.d_model, tgt_vocab)
        self.kg_loss_weight = getattr(args, 'kg_loss_weight', 0.1)

        # Contrastive Attention (optional)
        use_ca = getattr(args, 'use_contrastive_attention', False)
        if use_ca:
            self.contrastive_attn = ContrastiveAttention(
                d_model=args.d_model,
                d_fc = d_visual,
                pool_size=getattr(args, 'ca_pool_size', 100),
                num_agg_rounds=getattr(args, 'ca_num_rounds', 3),
                dropout=args.dropout,
            )
        else:
            self.contrastive_attn = None

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        if self.contrastive_attn is not None:
            att_feats = self.contrastive_attn(att_feats, self._cached_fc_feats)
        memory = self.model.encode(att_feats, att_masks)
        return fc_feats[..., :1], att_feats[..., :1], memory, att_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)
        if seq is not None:
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True
            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None
        return att_feats, seq, att_masks, seq_mask

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)
        if self.contrastive_attn is not None:
            att_feats = self.contrastive_attn(att_feats, fc_feats)
        kg_feats = self.kg_encoder(self.adj, fc_feats)
        out = self.model(att_feats, seq, att_masks, seq_mask, kg_feats)
        outputs = F.log_softmax(self.logit(out), dim=-1)
        return outputs

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
        if len(state) == 0:
            ys = it.unsqueeze(1)
            self._cached_kg_feats = self.kg_encoder(self.adj, self._cached_fc_feats)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        # Beam search batch expansion fix
        kg = self._cached_kg_feats
        if kg.size(0) != memory.size(0):
            beam = memory.size(0) // kg.size(0)
            kg = kg.unsqueeze(1).expand(-1, beam, -1, -1).reshape(
                memory.size(0), kg.size(1), kg.size(2))
            self._cached_kg_feats = kg
        out = self.model.decode(
            memory, mask, ys,
            subsequent_mask(ys.size(1)).to(memory.device),
            self._cached_kg_feats
        )
        return out[:, -1], [ys.unsqueeze(0)]

    def get_kg_labels(self, reports_text):
        labels = []
        for report in reports_text:
            labels.append(self.kg_builder.extract_labels_for_report(
                report, self.node_list, self.node2idx))
        return torch.FloatTensor(np.array(labels))