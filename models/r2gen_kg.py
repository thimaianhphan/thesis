# NOTE: Graph vocabulary is built dynamically from the training corpus using
# the hardcoded entity lists in modules/knowledge_graph.py. Checkpoints are
# tied to a specific node count. After any change to KnowledgeGraphBuilder
# (entity lists, co-occurrence threshold) you MUST retrain from scratch —
# old checkpoints will not load due to shape mismatches.

import torch
import torch.nn as nn
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.kg_encoder_decoder import KGEncoderDecoder, KGMultiLabelClassifier, KGAlignmentLoss

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
        self._cached_fc_feats = fc_feats  # used by classify_kg_nodes
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            self.encoder_decoder._cached_fc_feats = fc_feats
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output

    def forward_mimic_cxr(self, images, targets=None, mode='train'):
        att_feats, fc_feats = self.visual_extractor(images)
        self._cached_fc_feats = fc_feats  # used by classify_kg_nodes
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            self.encoder_decoder._cached_fc_feats = fc_feats
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output

    def classify_kg_nodes(self, images):
        # Reuse the fc_feats cached during the main forward() to skip a second
        # visual pass -- BUT ONLY when that cache is connected to the CURRENT
        # autograd graph. A cache left over from a no_grad / mode='sample'
        # (validation) forward is DETACHED; reusing it while training would
        # silently starve the visual extractor of gradients: the classifier and
        # head would still update, so the loss curve looks healthy and the bug is
        # invisible, yet the encoder never learns (any encoder retraining becomes
        # a no-op). When grad is globally disabled (eval), the cache is safe.
        # NOTE: the executed Stage-1 path (fresh model, no prior forward) and
        # Stage-2 path (forward overwrites the cache each iter before this read)
        # were already correct; this guard is defensive hardening for the
        # encoder-retraining objective and is behaviour-preserving in both.
        cached = getattr(self, '_cached_fc_feats', None)
        cache_usable = cached is not None and (cached.requires_grad or not torch.is_grad_enabled())
        if cache_usable:
            return self.kg_classifier(cached)
        # Recompute fresh, in-graph. Do NOT cache the result here: in the Stage-1
        # loop this method is called every iteration with no intervening
        # forward(), and caching a tensor whose graph is freed by backward()
        # would raise "backward through the graph a second time" next iteration.
        if self.args.dataset_name == 'iu_xray':
            _, fc_feats_0 = self.visual_extractor(images[:, 0])
            _, fc_feats_1 = self.visual_extractor(images[:, 1])
            fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        else:
            _, fc_feats = self.visual_extractor(images)
        return self.kg_classifier(fc_feats)
    
