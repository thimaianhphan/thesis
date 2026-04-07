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

KG node discovery:
  - BiomedCLIP text encoder types corpus-extracted terms into
    {anatomy, abnormal, normal} — replaces hardcoded word lists.

Contrastive Attention (optional, --use_contrastive_attention):
  - Normality pool built using kg_builder.is_normal_report()
    (BiomedCLIP-based) instead of keyword heuristics.
"""

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
                        choices=['resnet101', 'medsam'],
                        help="'resnet101' (d_vf=2048) or 'medsam' (d_vf=256)")
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True)
    parser.add_argument('--freeze_visual_extractor', action='store_true',
                        help='Freeze visual extractor backbone (useful for MedSAM).')

    # Transformer
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--d_vf', type=int, default=2048,
                        help='Patch feature dim. Set 256 for MedSAM, 2048 for ResNet.')
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
                        help='LR for visual extractor. Use 1e-5 for MedSAM.')
    parser.add_argument('--lr_ed', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--amsgrad', type=bool, default=True)

    # LR scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR')
    parser.add_argument('--step_size', type=int, default=50)
    parser.add_argument('--gamma', type=float, default=0.1)

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

    # BiomedCLIP KG typing
    parser.add_argument('--kg_min_term_freq', type=int, default=5,
                        help='Min document frequency for a corpus term to be a KG node.')
    parser.add_argument('--kg_max_nodes', type=int, default=150,
                        help='Hard cap on total KG nodes.')
    parser.add_argument('--biomedclip_device', type=str, default='cpu',
                        help="Device for BiomedCLIP during graph build / CA pool. "
                             "Use 'cuda' to speed up node typing on large corpora.")

    # ==================== Contrastive Attention args ====================
    parser.add_argument('--use_contrastive_attention', action='store_true',
                        help='Enable Contrastive Attention (Liu et al. ACL 2021).')
    parser.add_argument('--ca_pool_size', type=int, default=100,
                        help='Normality pool size. 100 for IU X-Ray, 500+ for MIMIC.')
    parser.add_argument('--ca_num_rounds', type=int, default=3,
                        help='Aggregate Attention rounds (paper uses 3).')

    args = parser.parse_args()

    # --- Consistency check ---
    if args.visual_extractor == 'medsam' and args.d_vf != 256:
        print(f"[WARNING] visual_extractor=medsam but d_vf={args.d_vf}. "
              f"Forcing d_vf=256.")
        args.d_vf = 256
    if args.visual_extractor == 'resnet101' and args.d_vf == 256:
        print(f"[WARNING] visual_extractor=resnet101 but d_vf=256. "
              f"Forcing d_vf=2048.")
        args.d_vf = 2048

    return args


def build_kg_optimizer(args, model):
    """Separate LR groups for visual extractor, KG encoder, CA, and the rest."""
    ve_ids = set(map(id, model.visual_extractor.parameters()))
    kg_ids = set(map(id, model.encoder_decoder.kg_encoder.parameters()))
    ca_ids = set()
    if model.encoder_decoder.contrastive_attn is not None:
        ca_ids = set(map(id, model.encoder_decoder.contrastive_attn.parameters()))

    special_ids = ve_ids | kg_ids | ca_ids
    ed_params = [p for p in model.parameters() if id(p) not in special_ids]

    param_groups = [
        {'params': list(model.visual_extractor.parameters()), 'lr': args.lr_ve},
        {'params': ed_params,                                  'lr': args.lr_ed},
        {'params': list(model.encoder_decoder.kg_encoder.parameters()), 'lr': args.lr_ed},
    ]
    if ca_ids:
        param_groups.append({
            'params': list(model.encoder_decoder.contrastive_attn.parameters()),
            'lr': args.lr_ed,
        })

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
            torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        print(f"[KG Stage 1] Epoch {epoch}/{args.kg_pretrain_epochs} "
              f"— Loss: {epoch_loss / max(n_batches, 1):.4f}")

    print("[KG Stage 1] Done — visual extractor now recognises clinical findings.")
    print("=" * 60)


def build_ca_pool(model, train_dataloader, args, device):
    """
    Build the Contrastive Attention normality pool.

    Uses kg_builder.is_normal_report() (BiomedCLIP-based) to identify normal
    training images instead of keyword heuristics.
    Ref: Liu et al. (ACL Findings 2021).
    """
    ca = model.encoder_decoder.contrastive_attn
    if ca is None:
        return

    print("=" * 60)
    print("[CA] Building normality pool")
    print(f"  Pool size        : {args.ca_pool_size}")
    print(f"  Agg rounds       : {args.ca_num_rounds}")
    print(f"  Normality scorer : BiomedCLIP (kg_builder.is_normal_report)")
    print(f"  Ref              : Liu et al. (ACL Findings 2021)")
    print("=" * 60)

    # Expose the kg_builder that was used to build the graph
    kg_builder = model.encoder_decoder.kg_builder

    ca.build_normality_pool(
        visual_extractor=model.visual_extractor,
        dataloader=train_dataloader,
        dataset_name=args.dataset_name,
        ann_path=args.ann_path,
        device=device,
        max_pool_size=args.ca_pool_size,
        kg_builder=kg_builder,          # ← BiomedCLIP-based normality scoring
    )
    print("[CA] Normality pool ready.")
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
    print(f"[Main] BiomedCLIP device: {args.biomedclip_device}")
    print(f"[Main] Contrastive Attn : {args.use_contrastive_attention}")

    # Tokenizer + dataloaders
    tokenizer = Tokenizer(args)
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader   = R2DataLoader(args, tokenizer, split='val',   shuffle=False)
    test_dataloader  = R2DataLoader(args, tokenizer, split='test',  shuffle=False)

    # ---- Build model ----
    # KnowledgeGraphBuilder inside KGEncoderDecoder.__init__ already runs
    # Phase 1 (corpus scan) + Phase 2 (BiomedCLIP typing) using args.biomedclip_device.
    # This happens once here before any GPU memory is allocated for training.
    model = R2GenKGModel(args, tokenizer)
    print(model)
    model = model.to(device)

    # ---- Build CA normality pool (before Stage 1 so pool uses clean backbone) ----
    if args.use_contrastive_attention:
        build_ca_pool(model, train_dataloader, args, device)

    # ---- Stage 1: KG pretraining ----
    pretrain_kg_classifier(model, train_dataloader, args, device)

    # ---- Stage 2: Report generation ----
    print("=" * 60)
    print("[KG Stage 2] Report generation training")
    print(f"  KG loss weight : {args.kg_loss_weight}")
    print("=" * 60)

    criterion    = compute_loss
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