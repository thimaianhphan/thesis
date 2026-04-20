"""
MedSAM Visual Extractor for R2Gen

Replaces the ResNet-101 visual_extractor.py with MedSAM's ViT-B image encoder.
Keeps the same interface: forward(images) -> (att_feats, fc_feats)

MedSAM encoder (wanglab/medsam-vit-base):
  - Input : [B, 3, 1024, 1024]  (MedSAM native) or [B, 3, 224, 224] (auto-resized)
  - After image encoder + neck:
      image_embeddings : [B, 256, 64, 64]   ← 64×64 spatial grid, 256-dim
  - We reshape to:
      att_feats        : [B, 4096, 256]      ← 64*64 = 4096 spatial tokens
      fc_feats         : [B, 256]            ← global average pool

  d_vf = 256  (set args.d_vf = 256 in your training script)

For IU X-Ray (dual image), R2GenKGModel calls this twice and concatenates:
  fc_feats  -> [B, 512]     (2 × 256)
  att_feats -> [B, 4096*2, 256] before the att_embed projection

Note on input resolution:
  MedSAM was trained at 1024×1024 but works well at 224×224 for feature
  extraction (the ViT just sees more coarse patches). We default to 224
  to keep batch throughput comparable to ResNet. Set args.image_size=1024
  for maximum quality if you have GPU memory.

Usage: replace args.visual_extractor = 'medsam' and set args.d_vf = 256.
       The existing visual_extractor.py checks for 'resnet101'; this file
       is loaded by a modified VisualExtractor dispatcher (see bottom of file).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SamModel, SamProcessor
 
 
MEDSAM_HF_ID = "wanglab/medsam-vit-base"
MEDSAM_D_VF = 256
 
 
class MedSAMVisualExtractor(nn.Module):
    """
    Visual feature extractor based on MedSAM's ViT-B image encoder.
 
    Calls vision_encoder DIRECTLY — bypasses get_image_embeddings() which
    enforces a 1024×1024 size check via SamProcessor. This lets us run at
    any resolution (224×224 default) without modification.
 
    MedSAM vision_encoder internals:
      patch_embed : Conv2d(3, 768, kernel=16, stride=16)
                    At 224×224 → (224/16)² = 196 patches  [B, 196, 768]
                    At 1024×1024 → 4096 patches            [B, 4096, 768]
      transformer : 12 ViT-B layers                        [B, P, 768]
      neck        : Conv2d 768→256 + LayerNorm             [B, 256, H/16, W/16]
 
    At 224×224:
      att_feats : [B, 196, 256]   (14×14 grid)
      fc_feats  : [B, 256]
 
    At 1024×1024:
      att_feats : [B, 4096, 256]  (64×64 grid)
      fc_feats  : [B, 256]
 
    d_vf = 256 in both cases. Set args.d_vf = 256.
    """
 
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.image_size = getattr(args, 'image_size', 224)
 
        pretrained = getattr(args, 'visual_extractor_pretrained', True)
        print(f"[MedSAM] Loading {MEDSAM_HF_ID} (pretrained={pretrained})...")
 
        full_model = SamModel.from_pretrained(MEDSAM_HF_ID)
 
        # Extract only the vision encoder (ViT + neck).
        # Discard prompt_encoder and mask_decoder — saves ~200MB.
        self.vision_encoder = full_model.vision_encoder
        del full_model
 
        if not pretrained:
            self.vision_encoder.apply(self._init_weights)
 
        frozen = getattr(args, 'freeze_visual_extractor', False)
        if frozen:
            for p in self.vision_encoder.parameters():
                p.requires_grad_(False)
            print("[MedSAM] Backbone frozen.")
        else:
            print("[MedSAM] Backbone trainable.")
 
        print(f"[MedSAM] Ready. image_size={self.image_size}, d_vf={MEDSAM_D_VF}")
 
    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
 
    def forward(self, images):
        """
        Args:
            images : [B, 3, H, W]  any resolution; resized internally if needed.
 
        Returns:
            att_feats : [B, N_patches, 256]  e.g. [B, 196, 256] at 224×224
            fc_feats  : [B, 256]
        """
        # Resize if necessary — purely in-model, no processor involved
        if images.shape[-2] != self.image_size or images.shape[-1] != self.image_size:
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False,
            )
 
        frozen = getattr(self.args, 'freeze_visual_extractor', False)
        with torch.set_grad_enabled(self.training and not frozen):
            # Call vision_encoder directly — no size validation, no processor.
            # Returns a BaseModelOutput; last_hidden_state is post-neck [B, 256, H/16, W/16]
            encoder_out = self.vision_encoder(pixel_values=images)
            # post-neck shape: [B, 256, image_size//16, image_size//16]
            feat_map = encoder_out.last_hidden_state   # [B, 256, h, w]
 
        # att_feats: flatten spatial dims → [B, h*w, 256]
        att_feats = feat_map.flatten(2).transpose(1, 2)   # [B, N, 256]
 
        # fc_feats: global average pool → [B, 256]
        fc_feats = feat_map.mean(dim=(2, 3))              # [B, 256]
 
        return att_feats, fc_feats

class ResNetVisualExtractor(nn.Module):
    def __init__(self, args):
        super(ResNetVisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, images):
        patch_feats = self.model(images)
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats

# =============================================================================
# Dispatcher: drop-in replacement for the original VisualExtractor
# =============================================================================

class VisualExtractor(nn.Module):
    """
    Unified visual extractor dispatcher.

    args.visual_extractor:
        'resnet101' (default) → original ResNet-101 extractor
        'medsam'              → MedSAM ViT-B extractor (this file)

    Interface is identical:  forward(images) → (att_feats, fc_feats)
    """

    def __init__(self, args):
        super().__init__()
        extractor_name = getattr(args, 'visual_extractor', 'resnet101')

        if extractor_name == 'medsam':
            self.extractor = MedSAMVisualExtractor(args)
            self.d_vf = MEDSAM_D_VF
        else:
            self.extractor = ResNetVisualExtractor(args)
            self.d_vf = getattr(args, 'd_vf', 2048)

    def forward(self, images):
        return self.extractor(images)