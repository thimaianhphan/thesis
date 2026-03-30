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
    
