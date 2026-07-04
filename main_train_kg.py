"""
Training script for R2Gen + Knowledge Graph.

Implements the two-stage training strategy from Zhang et al. (AAAI 2020):

Stage 1 (optional, --kg_pretrain_epochs > 0):
  - Train multi-label classifier to predict which KG nodes are present
  - This teaches the visual extractor to recognize clinical findings

Stage 2 (main training):
  - Train full report generation with KG integration
  - Loss = L_CE + lambda * L_KG_align
  - KG cross-attention in decoder provides clinical vocabulary guidance

Visual extractor:
  - 'resnet101' (default) : original R2Gen backbone, d_vf=2048
  - 'medsam'              : MedSAM ViT-B, d_vf=256
  - 'autoencoder'         : ConvAutoencoder encoder pretrained on IU X-Ray, d_vf=256
  - 'mae'                 : MAE-pretrained ViT-Small/16 (medical_mae, CheXpert+NIH), d_vf=384

KG node discovery:
  - Hardcoded anatomy/finding word lists (see modules/knowledge_graph.py).
    An earlier BiomedCLIP-based dynamic typing scheme was tried and reverted
    after it hurt report-generation quality.

Note: Contrastive Attention (Liu et al. ACL 2021) was tried and removed —
it also hurt report-generation quality.
"""

import functools
import torch
import argparse
import numpy as np
import json

from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from modules.loss import compute_loss
from models.r2gen_kg import R2GenKGModel


def parse_agrs():
    parser = argparse.ArgumentParser()

    # ==================== Original R2Gen args ====================
    parser.add_argument('--image_dir', type=str, default='data/iu_xray/images/')
    parser.add_argument('--ann_path', type=str, default='data/iu_xray/annotation.json')
    parser.add_argument('--dataset_name', type=str, default='iu_xray',
                        choices=['iu_xray', 'mimic_cxr'])
    parser.add_argument('--max_seq_length', type=int, default=60)
    parser.add_argument('--threshold', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)

    # Visual extractor
    parser.add_argument('--visual_extractor', type=str, default='resnet101',
                        choices=['resnet101', 'medsam', 'resnet50', 'autoencoder', 'mae'],
                        help="'resnet101' (d_vf=2048), 'medsam' (d_vf=256), 'autoencoder' (d_vf=256), or 'mae' (d_vf=384)")
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True)
    parser.add_argument('--autoencoder_ckpt', type=str, default='artifacts/ae_encoder.pth',
                        help='path to ae_encoder.pth (required when --visual_extractor autoencoder).')
    parser.add_argument('--mae_ckpt', type=str, default='artifacts/vit_small_mae.pth',
                        help='path to MAE-pretrained ViT-S/16 checkpoint (required when --visual_extractor mae).')
    parser.add_argument('--freeze_visual_extractor', action='store_true',
                        help='Freeze visual extractor backbone (useful for MedSAM/autoencoder/MAE).')

    # Transformer
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--d_vf', type=int, default=2048,
                        help='Patch feature dim. Set 256 for MedSAM/autoencoder, 384 for MAE, 2048 for ResNet.')
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--logit_layers', type=int, default=1)
    parser.add_argument('--bos_idx', type=int, default=0)
    parser.add_argument('--eos_idx', type=int, default=0)
    parser.add_argument('--pad_idx', type=int, default=0)
    parser.add_argument('--use_bn', type=int, default=0)
    parser.add_argument('--drop_prob_lm', type=float, default=0.5)

    # Relational Memory
    parser.add_argument('--rm_num_slots', type=int, default=3)
    parser.add_argument('--rm_num_heads', type=int, default=8)
    parser.add_argument('--rm_d_model', type=int, default=512)

    # Sampling / beam search
    parser.add_argument('--sample_method', type=str, default='beam_search')
    parser.add_argument('--beam_size', type=int, default=3)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sample_n', type=int, default=1)
    parser.add_argument('--group_size', type=int, default=1)
    parser.add_argument('--output_logsoftmax', type=int, default=1)
    parser.add_argument('--decoding_constraint', type=int, default=0)
    parser.add_argument('--block_trigrams', type=int, default=1)

    # Trainer
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='results/iu_xray_kg')
    parser.add_argument('--record_dir', type=str, default='records/')
    parser.add_argument('--save_period', type=int, default=1)
    parser.add_argument('--monitor_mode', type=str, default='max',
                        choices=['min', 'max'])
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4')
    parser.add_argument('--early_stop', type=int, default=50)

    # Optimisation
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--lr_ve', type=float, default=5e-5,
                        help='LR for visual extractor. Use 1e-5 for MedSAM/MAE.')
    parser.add_argument('--lr_ed', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--amsgrad', type=bool, default=True)

    # LR scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR')
    parser.add_argument('--step_size', type=int, default=50)
    parser.add_argument('--gamma', type=float, default=0.1)

    # Loss
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing epsilon (0 = off).')

    # Architecture switches
    parser.add_argument('--use_expert_memory', action='store_true',
                        help='Replace RelationalMemory with image-conditioned ExpertMemory.')
    parser.add_argument('--expert_query_causal_mean', action='store_true',
                        help='ExpertMemory queries experts with the causal running mean of '
                             'token embeddings instead of the raw per-token embedding.')

    # Misc
    parser.add_argument('--seed', type=int, default=9233)
    parser.add_argument('--resume', type=str,
                        help='Resume training from checkpoint.')

    # ==================== Knowledge Graph args ====================
    parser.add_argument('--kg_num_gcn_layers', type=int, default=1)
    parser.add_argument('--kg_gcn_alpha', type=float, default=0.2)
    parser.add_argument('--kg_loss_weight', type=float, default=0.1)
    parser.add_argument('--kg_pretrain_epochs', type=int, default=10)
    parser.add_argument('--kg_pretrain_lr', type=float, default=1e-4)
    parser.add_argument('--kg_co_occur_threshold', type=int, default=3)

    args = parser.parse_args()

    # --- Consistency check ---
    if args.visual_extractor in ('medsam', 'autoencoder') and args.d_vf != 256:
        print(f"[WARNING] visual_extractor={args.visual_extractor} but d_vf={args.d_vf}. "
              f"Forcing d_vf=256.")
        args.d_vf = 256
    if args.visual_extractor == 'mae' and args.d_vf != 384:
        print(f"[WARNING] visual_extractor=mae but d_vf={args.d_vf}. "
              f"Forcing d_vf=384.")
        args.d_vf = 384
    if args.visual_extractor == 'autoencoder' and not args.autoencoder_ckpt:
        raise ValueError("--visual_extractor autoencoder requires --autoencoder_ckpt <path to ae_encoder.pth>")
    if args.visual_extractor == 'mae' and not args.mae_ckpt:
        raise ValueError("--visual_extractor mae requires --mae_ckpt <path to MAE ViT-S/16 checkpoint>")
    if args.visual_extractor == 'resnet101' and args.d_vf == 256:
        print(f"[WARNING] visual_extractor=resnet101 but d_vf=256. "
              f"Forcing d_vf=2048.")
        args.d_vf = 2048

    return args


def build_kg_optimizer(args, model):
    """Separate LR groups for visual extractor, KG encoder, and the rest."""
    ve_ids = set(map(id, model.visual_extractor.parameters()))
    kg_ids = set(map(id, model.encoder_decoder.kg_encoder.parameters()))

    special_ids = ve_ids | kg_ids
    ed_params = [p for p in model.parameters() if id(p) not in special_ids]

    param_groups = [
        {'params': list(model.visual_extractor.parameters()), 'lr': args.lr_ve},
        {'params': ed_params,                                  'lr': args.lr_ed},
        {'params': list(model.encoder_decoder.kg_encoder.parameters()), 'lr': args.lr_ed},
    ]

    optimizer = getattr(torch.optim, args.optim)(
        param_groups,
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad,
    )
    return optimizer


def pretrain_kg_classifier(model, train_dataloader, args, device):
    """
    Stage 1: Multi-label classification pretraining.
    Ref: Zhang et al. (AAAI 2020).
    """
    if args.kg_pretrain_epochs <= 0:
        print("[KG Stage 1] Skipped (kg_pretrain_epochs=0)")
        return

    print("=" * 60)
    print("[KG Stage 1] Multi-label classification pretraining")
    print(f"  Epochs : {args.kg_pretrain_epochs}")
    print(f"  Ref    : Zhang et al. (AAAI 2020)")
    print("=" * 60)

    optimizer = torch.optim.Adam([
        {'params': model.visual_extractor.parameters(), 'lr': args.lr_ve},
        {'params': model.kg_classifier.parameters(),   'lr': args.kg_pretrain_lr},
    ], weight_decay=args.weight_decay)

    ann = json.loads(open(args.ann_path, 'r').read())
    train_reports = {ex['id']: ex['report'] for ex in ann['train']}

    model.train()
    for epoch in range(1, args.kg_pretrain_epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        for images_id, images, reports_ids, reports_masks in train_dataloader:
            images = images.to(device)
            batch_reports = [train_reports.get(iid, '') for iid in images_id]
            kg_labels = model.encoder_decoder.get_kg_labels(batch_reports).to(device)

            logits = model.classify_kg_nodes(images)
            loss = model.kg_classifier.get_loss(logits, kg_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        print(f"[KG Stage 1] Epoch {epoch}/{args.kg_pretrain_epochs} "
              f"— Loss: {epoch_loss / max(n_batches, 1):.4f}")

    print("[KG Stage 1] Done — visual extractor now recognises clinical findings.")
    print("=" * 60)


def main():
    args = parse_agrs()

    # Reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    device = torch.device(
        'cuda:0' if args.n_gpu > 0 and torch.cuda.is_available() else 'cpu'
    )
    print(f"[Main] Device: {device}")
    print(f"[Main] Visual extractor : {args.visual_extractor}  (d_vf={args.d_vf})")

    # Tokenizer + dataloaders
    tokenizer = Tokenizer(args)
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader   = R2DataLoader(args, tokenizer, split='val',   shuffle=False)
    test_dataloader  = R2DataLoader(args, tokenizer, split='test',  shuffle=False)

    # ---- Build model ----
    # KnowledgeGraphBuilder inside KGEncoderDecoder.__init__ scans the corpus
    # and types terms via the hardcoded entity lists in modules/knowledge_graph.py.
    # This happens once here before any GPU memory is allocated for training.
    model = R2GenKGModel(args, tokenizer)
    print(model)
    model = model.to(device)

    # ---- Stage 1: KG pretraining ----
    pretrain_kg_classifier(model, train_dataloader, args, device)

    # ---- Stage 2: Report generation ----
    print("=" * 60)
    print("[KG Stage 2] Report generation training")
    print(f"  KG loss weight : {args.kg_loss_weight}")
    print("=" * 60)

    criterion    = functools.partial(compute_loss, label_smoothing=args.label_smoothing)
    metrics      = compute_scores
    optimizer    = build_kg_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    trainer = Trainer(
        model, criterion, metrics, optimizer, args, lr_scheduler,
        train_dataloader, val_dataloader, test_dataloader,
    )
    trainer.train()


if __name__ == '__main__':
    main()