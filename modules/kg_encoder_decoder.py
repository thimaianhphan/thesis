"""
R2Gen + Knowledge Graph Integration

This module integrates the medical knowledge graph into R2Gen's architecture.
The KG is fused at two points following published approaches:

1. KG Cross-Attention in the decoder (KiUT, CVPR 2023)
   - Additional cross-attention layer between decoder hidden states and GCN-encoded KG nodes
   
2. KG Vocabulary Bias on output logits (Yang et al., MedIA 2022)
   - Directly boosts probability of clinical terms when KG context is relevant

Training follows the two-stage strategy from Zhang et al. (AAAI 2020):
- Stage 1: Multi-label KG node classification (pretraining)
- Stage 2: Full report generation with KG alignment loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .visual_extractor import VisualExtractor
from .encoder_decoder import (
    EncoderDecoder, Transformer, Encoder, Decoder, EncoderLayer, DecoderLayer,
    MultiHeadedAttention, PositionwiseFeedForward, PositionalEncoding,
    Embeddings, RelationalMemory, ConditionalSublayerConnection, LayerNorm,
    clones, subsequent_mask
)
from .att_model import pack_wrapper, AttModel
from .knowledge_graph import (
    KnowledgeGraphBuilder, KnowledgeGraphEncoder, KGCrossAttention,
    KGVocabularyBias, KGMultiLabelClassifier, KGAlignmentLoss
)

import copy
import math


# =============================================================================
# Modified Decoder Layer: adds KG cross-attention as 4th sublayer
# Ref: KiUT (Huang et al., CVPR 2023) — injects knowledge into decoder
# =============================================================================

class KGDecoderLayer(nn.Module):
    """
    Extended DecoderLayer with KG cross-attention.
    
    Original R2Gen decoder layer has 3 sublayers:
        1. Self-attention (with MCLN)
        2. Source attention over encoder output (with MCLN)
        3. Feed-forward (with MCLN)
    
    This adds a 4th sublayer after source attention:
        2.5. KG cross-attention (gated, with pre-norm)
    
    Following KiUT (CVPR 2023), the KG attention is injected between
    the visual cross-attention and the feed-forward layer.
    """
    
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout,
                 rm_num_slots, rm_d_model, num_heads=8):
        super(KGDecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        
        # Original 3 sublayers with MCLN
        self.sublayer = clones(
            ConditionalSublayerConnection(d_model, dropout, rm_num_slots, rm_d_model), 3
        )
        
        # NEW: KG cross-attention module (gated)
        self.kg_cross_attn = KGCrossAttention(d_model, num_heads, dropout)
        
    def forward(self, x, hidden_states, src_mask, tgt_mask, memory, kg_feats=None):
        """
        Args:
            x: Decoder input [B, L, D]
            hidden_states: Encoder output [B, S, D]
            src_mask: Source mask [B, 1, S]
            tgt_mask: Target mask [B, L, L]
            memory: Relational memory [B, rm_num_slots * rm_d_model]
            kg_feats: KG node representations [N, D] (optional)
        """
        m = hidden_states
        
        # Sublayer 1: Self-attention with MCLN
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask), memory)
        
        # Sublayer 2: Source (visual) attention with MCLN
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask), memory)
        
        # Sublayer 2.5 (NEW): KG cross-attention
        if kg_feats is not None:
            x = self.kg_cross_attn(x, kg_feats)
        
        # Sublayer 3: Feed-forward with MCLN
        return self.sublayer[2](x, self.feed_forward, memory)


class KGDecoder(nn.Module):
    """Decoder with KG-aware layers."""
    
    def __init__(self, layer, N):
        super(KGDecoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)
        
    def forward(self, x, hidden_states, src_mask, tgt_mask, memory, kg_feats=None):
        for layer in self.layers:
            x = layer(x, hidden_states, src_mask, tgt_mask, memory, kg_feats)
        return self.norm(x)


# =============================================================================
# Modified Transformer: passes KG features through decode path
# =============================================================================

class KGTransformer(nn.Module):
    """Transformer with KG-enhanced decoder."""
    
    def __init__(self, encoder, decoder, src_embed, tgt_embed, rm):
        super(KGTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.rm = rm
        
    def forward(self, src, tgt, src_mask, tgt_mask, kg_feats=None):
        return self.decode(
            self.encode(src, src_mask), src_mask, tgt, tgt_mask, kg_feats
        )
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, hidden_states, src_mask, tgt, tgt_mask, kg_feats=None):
        memory = self.rm.init_memory(hidden_states.size(0)).to(hidden_states)
        memory = self.rm(self.tgt_embed(tgt), memory)
        return self.decoder(
            self.tgt_embed(tgt), hidden_states, src_mask, tgt_mask, memory, kg_feats
        )


# =============================================================================
# KG-Enhanced EncoderDecoder (replaces original EncoderDecoder)
# =============================================================================

class KGEncoderDecoder(AttModel):
    """
    R2Gen's EncoderDecoder with integrated Knowledge Graph.
    
    Changes from original:
    1. DecoderLayer → KGDecoderLayer (adds KG cross-attention)
    2. Transformer → KGTransformer (passes kg_feats through decode)
    3. Added KGVocabularyBias on output logits
    4. KG features computed once per forward pass and cached
    """
    
    def make_model(self, tgt_vocab, kg_encoder=None, kg_vocab_bias=None):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        rm = RelationalMemory(
            num_slots=self.rm_num_slots, d_model=self.rm_d_model,
            num_heads=self.rm_num_heads
        )
        
        # Use KG-enhanced decoder layer
        model = KGTransformer(
            Encoder(
                EncoderLayer(self.d_model, c(attn), c(ff), self.dropout),
                self.num_layers
            ),
            KGDecoder(
                KGDecoderLayer(
                    self.d_model, c(attn), c(attn), c(ff), self.dropout,
                    self.rm_num_slots, self.rm_d_model, self.num_heads
                ),
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
        super(KGEncoderDecoder, self).__init__(args, tokenizer)
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
        kg_builder = KnowledgeGraphBuilder(args.ann_path, args.dataset_name)
        node_list, node_types, adjacency, node2idx = kg_builder.build(split='train')
        
        self.node_list = node_list
        self.node_types = node_types
        self.node2idx = node2idx
        self.kg_builder = kg_builder
        self.num_kg_nodes = len(node_list)
        
        # Store adjacency as buffer (not a parameter)
        self.register_buffer('adj', torch.FloatTensor(adjacency))
        
        # KG encoder (GCN)
        self.kg_encoder = KnowledgeGraphEncoder(
            num_nodes=self.num_kg_nodes,
            node_types=node_types,
            d_model=args.d_model,
            num_gcn_layers=getattr(args, 'kg_num_gcn_layers', 2),
            dropout=args.dropout
        )
        
        # KG vocabulary bias
        self.kg_vocab_bias = KGVocabularyBias(
            d_model=args.d_model,
            num_nodes=self.num_kg_nodes,
            vocab_size=self.vocab_size,
            node_list=node_list,
            tokenizer=tokenizer
        )
        
        # Build modified model
        self.model = self.make_model(tgt_vocab)
        self.logit = nn.Linear(args.d_model, tgt_vocab)
        
        # KG alignment loss weight
        self.kg_loss_weight = getattr(args, 'kg_loss_weight', 0.1)
        
    def _encode_kg(self):
        """Encode KG nodes through GCN. Called once per forward pass."""
        return self.kg_encoder(self.adj)  # [N, d_model]
    
    def init_hidden(self, bsz):
        return []
    
    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(
            att_feats, att_masks
        )
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
        """Training forward pass with KG integration."""
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(
            att_feats, att_masks, seq
        )
        
        # Encode KG (once per batch)
        kg_feats = self._encode_kg()  # [N, d_model]
        
        # Forward through KG-enhanced transformer
        out = self.model(att_feats, seq, att_masks, seq_mask, kg_feats)
        
        # Get base logits
        logits = self.logit(out)
        
        # Apply KG vocabulary bias
        logits = self.kg_vocab_bias(out, kg_feats, logits)
        
        outputs = F.log_softmax(logits, dim=-1)
        return outputs
    
    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
        """Inference step-by-step decoding with KG."""
        if len(state) == 0:
            ys = it.unsqueeze(1)
            # Encode KG at start of decoding
            self._cached_kg_feats = self._encode_kg()
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        
        out = self.model.decode(
            memory, mask, ys,
            subsequent_mask(ys.size(1)).to(memory.device),
            self._cached_kg_feats
        )
        
        return out[:, -1], [ys.unsqueeze(0)]
    
    def get_kg_labels(self, reports_text):
        """
        Extract multi-label KG ground truth from report texts.
        Used for KG alignment loss during training.
        
        Args:
            reports_text: List of report strings
            
        Returns:
            labels: Tensor [B, num_kg_nodes]
        """
        labels = []
        for report in reports_text:
            label = self.kg_builder.extract_labels_for_report(
                report, self.node_list, self.node2idx
            )
            labels.append(label)
        return torch.FloatTensor(np.array(labels))
