"""
R2Gen + DCG-style Knowledge Graph Fusion

Data flow:
  Image → ResNet-101 (224×224) → att_feats [B, 98, 2048] → att_embed → [B, 98, 512]
                                                                           ↓
                                                                  R2Gen Encoder (self-attn)
                                                                           ↓
                                                                  encoder_out [B, 98, 512]
                                                                           ↓
  Entity text → distilGPT2 (offline) → [N, 768] → proj → [N, 512]         ↓
                                                      ↓                    ↓
                                              2-layer GCN(adj)             ↓
                                                      ↓                    ↓
                                              node_feats [N, 512]          ↓
                                                      ↓                    ↓
                                            DCG Bidirectional Cross-Attention:
                                              i2g: Q=encoder_out, K/V=node_feats
                                              g2i: Q=node_feats, K/V=encoder_out
                                                      ↓
                                              concat [B, N+98, 512]
                                                      ↓
                                      R2Gen Decoder (RM + MCLN, standard, unchanged)
                                                      ↓
                                              logits → report

Ref: DCG, Liang et al. "Divide and Conquer" (ACM MM 2024)
Ref: R2Gen, Chen et al. (EMNLP 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from .visual_extractor import VisualExtractor
from .encoder_decoder import (
    Encoder, Decoder, EncoderLayer, DecoderLayer,
    MultiHeadedAttention, PositionwiseFeedForward, PositionalEncoding,
    Embeddings, RelationalMemory, LayerNorm, subsequent_mask
)
from .att_model import pack_wrapper, AttModel
from .knowledge_graph import (
    KnowledgeGraphBuilder, KnowledgeGraphEncoder, DCGFusion,
    extract_node_features_gpt2,
)
from .contrastive_attention import ContrastiveAttention


# =============================================================================
# DCG-enhanced Transformer: Encoder → DCG Fusion → Standard Decoder
# =============================================================================

class DCGTransformer(nn.Module):
    """
    R2Gen Transformer with DCG fusion between encoder and decoder.
    Decoder is STANDARD R2Gen (DecoderLayer with RM + MCLN).
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, rm, dcg_fusion):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.rm = rm
        self.dcg_fusion = dcg_fusion

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def fuse(self, encoder_out, kg_feats):
        """DCG bidirectional cross-attention after encoder."""
        return self.dcg_fusion(encoder_out, kg_feats)

    def decode(self, fused_feats, fused_mask, tgt, tgt_mask):
        """Standard R2Gen decode on fused features."""
        memory = self.rm.init_memory(fused_feats.size(0)).to(fused_feats)
        memory = self.rm(self.tgt_embed(tgt), memory)
        return self.decoder(
            self.tgt_embed(tgt), fused_feats, fused_mask, tgt_mask, memory
        )

    def forward(self, src, tgt, src_mask, tgt_mask, kg_feats):
        encoder_out = self.encode(src, src_mask)
        fused = self.fuse(encoder_out, kg_feats)
        B = fused.size(0)
        fused_mask = fused.new_ones(B, 1, fused.size(1)).long()
        return self.decode(fused, fused_mask, tgt, tgt_mask)


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
        dcg_fusion = DCGFusion(
            self.d_model, num_heads=self.num_heads, dropout=self.dropout
        )
        # Standard R2Gen decoder — NO modifications
        model = DCGTransformer(
            Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers),
            Decoder(
                DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout,
                             self.rm_num_slots, self.rm_d_model),
                self.num_layers
            ),
            lambda x: x,
            nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position)),
            rm,
            dcg_fusion
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

        # Extract node features with distilGPT2 (offline, cached)
        cache_dir = os.path.dirname(args.ann_path) if hasattr(args, 'ann_path') else '.'
        cache_path = os.path.join(cache_dir, f'node_features_gpt2_{args.dataset_name}.pt')
        node_feat_init = extract_node_features_gpt2(
            node_list, cache_path=cache_path, device='cpu'
        )

        # KG encoder: 2-layer GCN with distilGPT2 init
        self.kg_encoder = KnowledgeGraphEncoder(
            num_nodes=self.num_kg_nodes,
            node_types=node_types,
            d_model=args.d_model,
            node_feat_init=node_feat_init,
            gcn_hidden=getattr(args, 'kg_gcn_hidden', 128),
            dropout=args.dropout,
        )

        self.model = self.make_model(tgt_vocab)
        self.logit = nn.Linear(args.d_model, tgt_vocab)

        # Contrastive Attention (optional)
        use_ca = getattr(args, 'use_contrastive_attention', False)
        if use_ca:
            self.contrastive_attn = ContrastiveAttention(
                d_model=args.d_model,
                pool_size=getattr(args, 'ca_pool_size', 100),
                num_agg_rounds=getattr(args, 'ca_num_rounds', 3),
                dropout=args.dropout,
            )
        else:
            self.contrastive_attn = None

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        """Inference: encode → fuse → cache for decoder."""
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)

        if self.contrastive_attn is not None:
            att_feats = self.contrastive_attn(att_feats, self._cached_fc_feats)

        encoder_out = self.model.encode(att_feats, att_masks)

        # GCN (no image conditioning — DCG style)
        kg_feats = self.kg_encoder(self.adj)  # [N, D]
        B = encoder_out.size(0)
        kg_feats = kg_feats.unsqueeze(0).expand(B, -1, -1)  # [B, N, D]

        # DCG fusion
        fused = self.model.fuse(encoder_out, kg_feats)  # [B, N+S, D]
        fused_mask = fused.new_ones(B, 1, fused.size(1)).long()

        return fc_feats[..., :1], att_feats[..., :1], fused, fused_mask

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
        """Training forward."""
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)

        if self.contrastive_attn is not None:
            att_feats = self.contrastive_attn(att_feats, fc_feats)

        # GCN (no image conditioning)
        kg_feats = self.kg_encoder(self.adj)  # [N, D]
        B = att_feats.size(0)
        kg_feats = kg_feats.unsqueeze(0).expand(B, -1, -1)  # [B, N, D]

        out = self.model(att_feats, seq, att_masks, seq_mask, kg_feats)
        outputs = F.log_softmax(self.logit(out), dim=-1)
        return outputs

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
        """Inference step — memory is fused features from _prepare_feature."""
        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)

        out = self.model.decode(
            memory, mask, ys,
            subsequent_mask(ys.size(1)).to(memory.device)
        )
        return out[:, -1], [ys.unsqueeze(0)]

    def get_kg_labels(self, reports_text):
        labels = []
        for report in reports_text:
            labels.append(self.kg_builder.extract_labels_for_report(
                report, self.node_list, self.node2idx))
        return torch.FloatTensor(np.array(labels))