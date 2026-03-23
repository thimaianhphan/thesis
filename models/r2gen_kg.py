 """
    R2Gen + Knowledge Graph model.
    
    This replaces R2GenModel with KG-integrated version.
    
    Architecture changes:
    - KGEncoderDecoder replaces EncoderDecoder
    - KG features from GCN are passed to every decoder layer
    - Output logits are biased toward clinical terms via KGVocabularyBias
    
    Paper references for each component:
    - Base architecture: Chen et al. "R2Gen" (EMNLP 2020)
    - KG construction: Zhang et al. (AAAI 2020)
    - GCN encoding: Kipf & Welling (ICLR 2017)
    - Normal/abnormal separation: Liang et al. "DCG" (ACM MM 2024)
    - KG cross-attention in decoder: Huang et al. "KiUT" (CVPR 2023)
    - Gated knowledge distillation: Liu et al. "PPKED" (CVPR 2021)
    - Two-stage training: Zhang et al. (AAAI 2020)
    
The data flow is now:
  Image → ResNet → fc_feats ─────────────────────→ KG Encoder → kg_feats [B,N,D]
                 → att_feats → Transformer Encoder → encoder_out
                                                            ↓
  Decoder: self_attn → visual_cross_attn → KG_cross_attn(kg_feats) → FFN → logit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math

from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import (
    EncoderDecoder, Encoder, Decoder, EncoderLayer, DecoderLayer,
    MultiHeadedAttention, PositionwiseFeedForward, PositionalEncoding,
    Embeddings, RelationalMemory, ConditionalSublayerConnection, LayerNorm,
    clones, subsequent_mask
)
from modules.att_model import pack_wrapper, AttModel
from modules.knowledge_graph import (
    KnowledgeGraphBuilder, KnowledgeGraphEncoder, KGCrossAttention,
    KGMultiLabelClassifier, KGAlignmentLoss
)


# =============================================================================
# Modified Decoder Layer: KG cross-attention between visual attn and FFN
# =============================================================================

class KGDecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout,
                 rm_num_slots, rm_d_model, num_heads=4):
        super().__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(
            ConditionalSublayerConnection(d_model, dropout, rm_num_slots, rm_d_model), 3
        )
        self.kg_cross_attn = KGCrossAttention(d_model, num_heads, dropout)
        
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
    """
    R2Gen EncoderDecoder with KG integration.
    
    Critical difference from v1: fc_feats is passed to KG encoder
    so that kg_feats are per-image, not static.
    """
    
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
        kg_builder = KnowledgeGraphBuilder(args.ann_path, args.dataset_name)
        node_list, node_types, adjacency, node2idx = kg_builder.build(split='train')
        
        self.node_list = node_list
        self.node_types = node_types
        self.node2idx = node2idx
        self.kg_builder = kg_builder
        self.num_kg_nodes = len(node_list)
        self.register_buffer('adj', torch.FloatTensor(adjacency))
        
        # Visual feat dim depends on dataset
        # IU X-Ray: 2 views concatenated → d_vf * 2
        # MIMIC-CXR: single view → d_vf
        d_visual = args.d_vf * (2 if args.dataset_name == 'iu_xray' else 1)
        
        # KG encoder (now takes visual features)
        self.kg_encoder = KnowledgeGraphEncoder(
            num_nodes=self.num_kg_nodes,
            node_types=node_types,
            d_model=args.d_model,
            d_visual=d_visual,
            num_gcn_layers=getattr(args, 'kg_num_gcn_layers', 1),
            dropout=args.dropout,
            gcn_residual_alpha=getattr(args, 'kg_gcn_alpha', 0.2),
        )
        
        self.model = self.make_model(tgt_vocab)
        self.logit = nn.Linear(args.d_model, tgt_vocab)
        self.kg_loss_weight = getattr(args, 'kg_loss_weight', 0.1)
    
    def init_hidden(self, bsz):
        return []
    
    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
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
        """Training forward — fc_feats now flows into KG encoder."""
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)
        
        # KG encoding conditioned on visual features (THE FIX)
        kg_feats = self.kg_encoder(self.adj, fc_feats)  # [B, N, d_model]
        
        out = self.model(att_feats, seq, att_masks, seq_mask, kg_feats)
        outputs = F.log_softmax(self.logit(out), dim=-1)
        return outputs
    
    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
        """Inference — KG feats cached at first step, conditioned on fc_feats."""
        if len(state) == 0:
            ys = it.unsqueeze(1)
            # fc_feats_ph is a placeholder here; we need the real fc_feats.
            # They're cached in self._cached_fc_feats by the calling code.
            self._cached_kg_feats = self.kg_encoder(self.adj, self._cached_fc_feats)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        
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


# =============================================================================
# Complete R2Gen+KG Model
# =============================================================================

class R2GenKGModel(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = KGEncoderDecoder(args, tokenizer)
        
        d_visual = args.d_vf * (2 if args.dataset_name == 'iu_xray' else 1)
        self.kg_classifier = KGMultiLabelClassifier(
            visual_feat_dim=d_visual,
            num_nodes=self.encoder_decoder.num_kg_nodes,
            d_model=args.d_model
        )
        self.kg_align_loss = KGAlignmentLoss()
        
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr
    
    def __str__(self):
        params = sum([np.prod(p.size()) for p in self.parameters() if p.requires_grad])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
    
    def forward_iu_xray(self, images, targets=None, mode='train'):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            # Cache fc_feats for KG encoding during beam search
            self.encoder_decoder._cached_fc_feats = fc_feats
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output
    
    def forward_mimic_cxr(self, images, targets=None, mode='train'):
        att_feats, fc_feats = self.visual_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            self.encoder_decoder._cached_fc_feats = fc_feats
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output
    
    def classify_kg_nodes(self, images):
        if self.args.dataset_name == 'iu_xray':
            _, fc_feats_0 = self.visual_extractor(images[:, 0])
            _, fc_feats_1 = self.visual_extractor(images[:, 1])
            fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        else:
            _, fc_feats = self.visual_extractor(images)
        return self.kg_classifier(fc_feats)
    
