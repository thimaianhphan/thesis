"""
Visual Extractors for R2Gen

Alternative visual backbones to the original ResNet-101 visual_extractor.py:
MedSAM's ViT-B, a self-supervised ConvAutoencoder, and a MAE-pretrained
ViT-Small/16. All keep the same interface: forward(images) -> (att_feats, fc_feats)

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

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import SamModel, SamProcessor
from timm.models.vision_transformer import VisionTransformer

from modules.autoencoder import ConvEncoder


MEDSAM_HF_ID = "wanglab/medsam-vit-base"
MEDSAM_D_VF = 256
AE_D_VF = 256

# MAE-pretrained ViT-Small/16 (facebookresearch/mae format, checkpoint trained
# on CheXpert + NIH ChestX-ray14 — see lambert-x/medical_mae). The checkpoint's
# 'model' dict holds both encoder and decoder weights from MAE pretraining;
# only the encoder half is used here for feature extraction.
MAE_D_VF = 384
MAE_IMG_SIZE = 224
MAE_PATCH_SIZE = 16
MAE_EMBED_DIM = 384
MAE_DEPTH = 12
MAE_NUM_HEADS = 6


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


class AutoencoderVisualExtractor(nn.Module):
    """
    Visual feature extractor using the encoder half of a ConvAutoencoder
    (modules/autoencoder.py) pretrained self-supervised on IU X-Ray
    (see train_ae_iu_xray.ipynb).

    ConvEncoder internals:
        4x [Conv2d(k3,s2,p1) -> BatchNorm -> ReLU], 224x224 -> 14x14, channels 3->32->64->128->256

    At 224x224:
        att_feats : [B, 196, 256]  (14x14 grid)
        fc_feats  : [B, 256]

    d_vf = 256. Set args.d_vf = 256.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        ckpt_path = getattr(args, 'autoencoder_ckpt', None)
        if not ckpt_path:
            raise ValueError(
                "visual_extractor='autoencoder' requires --autoencoder_ckpt <path to ae_encoder.pth>"
            )

        self.encoder = ConvEncoder()
        print(f"[Autoencoder] Loading encoder weights from {ckpt_path}...")
        state_dict = torch.load(ckpt_path, map_location='cpu')
        self.encoder.load_state_dict(state_dict, strict=True)

        frozen = getattr(args, 'freeze_visual_extractor', False)
        if frozen:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            print("[Autoencoder] Encoder frozen.")
        else:
            print("[Autoencoder] Encoder trainable.")

    def forward(self, images):
        """
        Args:
            images : [B, 3, 224, 224]  ImageNet-normalized.

        Returns:
            att_feats : [B, 196, 256]
            fc_feats  : [B, 256]
        """
        frozen = getattr(self.args, 'freeze_visual_extractor', False)
        with torch.set_grad_enabled(self.training and not frozen):
            feat_map = self.encoder(images)  # [B, 256, 14, 14]

        att_feats = feat_map.flatten(2).transpose(1, 2)  # [B, 196, 256]
        fc_feats = feat_map.mean(dim=(2, 3))              # [B, 256]

        return att_feats, fc_feats


class MAEVisualExtractor(nn.Module):
    """
    Visual feature extractor using a MAE-pretrained ViT-Small/16 encoder.

    Checkpoint format: standard facebookresearch/mae pretraining output
    (model='mae_vit_small_patch16_dec512d2b'), e.g. the medical_mae checkpoint
    pretrained self-supervised on CheXpert + NIH ChestX-ray14. The checkpoint's
    'model' dict contains both encoder and decoder weights; decoder_*/mask_token
    keys are dropped here since the decoder is only used for MAE's pixel
    reconstruction pretraining objective, not for feature extraction.

    ViT-S/16 encoder: embed_dim=384, depth=12, heads=6, mlp_ratio=4, patch=16.
    At 224x224 -> 14x14=196 patches + 1 cls token.

    At 224x224:
        att_feats : [B, 196, 384]  (14x14 grid, cls token dropped)
        fc_feats  : [B, 384]       (mean over patch tokens)

    d_vf = 384. Set args.d_vf = 384.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        ckpt_path = getattr(args, 'mae_ckpt', None)
        if not ckpt_path:
            raise ValueError(
                "visual_extractor='mae' requires --mae_ckpt <path to MAE ViT-S/16 checkpoint>"
            )

        self.vit = VisionTransformer(
            img_size=MAE_IMG_SIZE,
            patch_size=MAE_PATCH_SIZE,
            embed_dim=MAE_EMBED_DIM,
            depth=MAE_DEPTH,
            num_heads=MAE_NUM_HEADS,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=0,
        )

        print(f"[MAE] Loading ViT-S/16 encoder weights from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt

        encoder_state = {
            k: v for k, v in state_dict.items()
            if not k.startswith('decoder_') and k != 'mask_token'
        }

        missing, unexpected = self.vit.load_state_dict(encoder_state, strict=False)
        if missing:
            print(f"[MAE] Missing keys (randomly initialized): {missing}")
        if unexpected:
            print(f"[MAE] Unexpected keys (ignored): {unexpected}")

        frozen = getattr(args, 'freeze_visual_extractor', False)
        if frozen:
            for p in self.vit.parameters():
                p.requires_grad_(False)
            print("[MAE] Backbone frozen.")
        else:
            print("[MAE] Backbone trainable.")

        print(f"[MAE] Ready. image_size={MAE_IMG_SIZE}, d_vf={MAE_D_VF}")

    def forward(self, images):
        """
        Args:
            images : [B, 3, H, W]  resized to 224x224 internally if needed.

        Returns:
            att_feats : [B, 196, 384]
            fc_feats  : [B, 384]
        """
        if images.shape[-2] != MAE_IMG_SIZE or images.shape[-1] != MAE_IMG_SIZE:
            images = F.interpolate(
                images,
                size=(MAE_IMG_SIZE, MAE_IMG_SIZE),
                mode='bilinear',
                align_corners=False,
            )

        frozen = getattr(self.args, 'freeze_visual_extractor', False)
        with torch.set_grad_enabled(self.training and not frozen):
            tokens = self.vit.forward_features(images)  # [B, 197, 384] incl. cls token

        patch_tokens = tokens[:, 1:, :]        # drop cls token -> [B, 196, 384]
        att_feats = patch_tokens
        fc_feats = patch_tokens.mean(dim=1)

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
        'autoencoder'         → ConvAutoencoder encoder pretrained on IU X-Ray (this file)
        'mae'                 → MAE-pretrained ViT-Small/16 extractor (this file)

    Interface is identical:  forward(images) → (att_feats, fc_feats)
    """

    def __init__(self, args):
        super().__init__()
        extractor_name = getattr(args, 'visual_extractor', 'resnet101')

        if extractor_name == 'medsam':
            self.extractor = MedSAMVisualExtractor(args)
            self.d_vf = MEDSAM_D_VF
        elif extractor_name == 'autoencoder':
            self.extractor = AutoencoderVisualExtractor(args)
            self.d_vf = AE_D_VF
        elif extractor_name == 'mae':
            self.extractor = MAEVisualExtractor(args)
            self.d_vf = MAE_D_VF
        else:
            self.extractor = ResNetVisualExtractor(args)
            self.d_vf = getattr(args, 'd_vf', 2048)

    def forward(self, images):
        return self.extractor(images)