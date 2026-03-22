import torch
import torch.nn as nn
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import KGEncoderDecoder, KGMultiLabelClassifier, KGAlignmentLoss

class R2GenKGModel(nn.Module):
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
    """
    
    def __init__(self, args, tokenizer):
        super(R2GenKGModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = KGEncoderDecoder(args, tokenizer)
        
        # Multi-label classifier for Stage 1 pretraining
        # (Zhang et al., AAAI 2020)
        self.kg_classifier = KGMultiLabelClassifier(
            visual_feat_dim=args.d_vf * (2 if args.dataset_name == 'iu_xray' else 1),
            num_nodes=self.encoder_decoder.num_kg_nodes,
            d_model=args.d_model
        )
        
        # KG alignment loss
        self.kg_align_loss = KGAlignmentLoss()
        
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr
    
    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
    
    def forward_iu_xray(self, images, targets=None, mode='train'):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output
    
    def forward_mimic_cxr(self, images, targets=None, mode='train'):
        att_feats, fc_feats = self.visual_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output
    
    def classify_kg_nodes(self, images):
        """
        Stage 1: Multi-label classification for KG nodes.
        
        Args:
            images: Input images [B, ...] 
        Returns:
            logits: [B, num_kg_nodes]
        """
        if self.args.dataset_name == 'iu_xray':
            _, fc_feats_0 = self.visual_extractor(images[:, 0])
            _, fc_feats_1 = self.visual_extractor(images[:, 1])
            fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        else:
            _, fc_feats = self.visual_extractor(images)
        return self.kg_classifier(fc_feats)