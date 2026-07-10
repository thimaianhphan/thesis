"""
Part B — Combined-objective retraining of the ConvAE encoder.

This is NOT "repairing an autoencoder". It CHANGES the pretraining objective from
"reconstruct appearance" to "predict findings, with reconstruction as a light
regularizer", so linearly-separable clinical structure is forced into the 256-d
bottleneck while preserving the from-scratch conv-encoder contribution.

    L = lambda_recon * MSE(decoder(z), image)  +  lambda_cls * FocalBCE(head(z), findings)

  - Encoder : ConvEncoder (modules/autoencoder.py), WARM-STARTED from ae_encoder.pth
  - Decoder : ConvDecoder, for the reconstruction regularizer (fresh unless
              --decoder_ckpt given)
  - Head    : GAP(z) -> Linear(256, n_findings)   [+ view embedding, see below]
  - FocalBCE: Lin et al., ICCV 2017 — handles finding-class imbalance and stops the
              head collapsing to "all negative".

Weights: lambda_recon=0.1, lambda_cls=1.0 (both CLI). FALLBACK: if the post-retrain
probe AUC stays weak, push --lambda_recon to 0.01-0.05 — the MSE exists only to keep
reconstructions from degenerating (preserve the reconstruction story); it must not
steer the representation.

View handling (Phase 0.2): findings labels are per-study, but IU X-ray has 2 views.
We pass a VIEW FLAG into the model as a learned embedding ADDED to the pooled
bottleneck feature before the classification head (--view_handling flag). The head
can then account for view-specific appearance so the classification loss does not
push the ENCODER to spend a latent axis distinguishing frontal/lateral. The
reconstruction path uses the raw bottleneck, unchanged. --view_handling none
disables it.

Labels: --labels negaware (default) uses the negation-aware findings labels;
Phase 0.1 found ~71% of RAW abnormal positives are negation artifacts, which would
teach the head to fire on negated findings. --labels raw reproduces the current
Stage-1 target.

HANDOFF: real training loads real images + real ckpt + wants a GPU -> run on the
remote machine. Locally only `--smoke` (synthetic tensors, CPU) is run; it does one
forward+backward and asserts both loss terms are finite & non-zero and encoder conv
grads are non-zero, then STOPS. No training is run here.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.abspath(__file__))
for p in (_ROOT, os.path.join(_ROOT, 'diagnostics')):
    if p not in sys.path:
        sys.path.insert(0, p)

from modules.autoencoder import ConvEncoder, ConvDecoder


# ----------------------------- Focal BCE -------------------------------------
class FocalBCEWithLogits(nn.Module):
    """Multi-label focal loss (Lin et al., ICCV 2017) on independent sigmoids."""

    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce * (1 - p_t).pow(self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


# ----------------------------- Model -----------------------------------------
class CombinedAE(nn.Module):
    def __init__(self, n_findings, view_handling='flag', n_views=2):
        super().__init__()
        self.encoder = ConvEncoder()
        self.decoder = ConvDecoder()
        self.head = nn.Linear(256, n_findings)
        self.view_handling = view_handling
        if view_handling == 'flag':
            self.view_embed = nn.Embedding(n_views, 256)
            nn.init.zeros_(self.view_embed.weight)  # start as no-op
        else:
            self.view_embed = None

    def forward(self, images, view_ids=None):
        z = self.encoder(images)                 # [B, 256, 14, 14]
        recon = self.decoder(z)                  # [B, 3, 224, 224]
        pooled = z.mean(dim=(2, 3))              # [B, 256]  GAP bottleneck
        h = pooled
        if self.view_embed is not None and view_ids is not None:
            h = pooled + self.view_embed(view_ids)
        logits = self.head(h)                    # [B, n_findings]
        return recon, logits, z


# ----------------------------- Data (real run) -------------------------------
class IuxrayPerImage(torch.utils.data.Dataset):
    """One sample per image: (image_tensor, view_id, label_vector). Labels are the
    per-study findings vector (same for both views of a study)."""

    def __init__(self, ann_path, image_dir, split, node_list, node2idx, labeler, transform):
        from PIL import Image
        self._Image = Image
        ann = json.loads(open(ann_path, 'r', encoding='utf-8').read())
        self.items = []
        for ex in ann[split]:
            label = labeler(ex['report'])
            for v, rp in enumerate(ex['image_path']):
                self.items.append((os.path.join(image_dir, rp), v, label))
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, v, label = self.items[i]
        img = self._Image.open(path).convert('RGB')
        return self.transform(img), v, torch.from_numpy(label)


def build_labeler(ann_path, kind, co_occur):
    """Return (node_list, node2idx, labeler_fn). Raw labels use the real builder."""
    import kg_labels as K
    ann = json.loads(open(ann_path, 'r', encoding='utf-8').read())
    try:
        from modules.knowledge_graph import KnowledgeGraphBuilder
        kgb = KnowledgeGraphBuilder(ann_path, 'iu_xray', co_occur_threshold=co_occur)
        node_list, node_types, _, node2idx = kgb.build(split='train')
        raw = lambda rep: kgb.extract_labels_for_report(rep, node_list, node2idx)
    except Exception as e:
        print(f"[warn] mirrored builder (real import failed: {e})")
        node_list, node_types, node2idx = K.build_nodes(ann, 'train')
        raw = lambda rep: K.labels_for_report(rep, node_list, node2idx)
    if kind == 'raw':
        return node_list, node2idx, raw
    neg = lambda rep: K.labels_for_report_negaware(rep, node_list, node2idx)
    return node_list, node2idx, neg


def build_train_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def train(args):
    device = torch.device('cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')
    node_list, node2idx, labeler = build_labeler(args.ann_path, args.labels, args.kg_co_occur_threshold)
    n_findings = len(node_list)
    print(f"[train] n_findings={n_findings} labels={args.labels} view_handling={args.view_handling} device={device}")

    model = CombinedAE(n_findings, args.view_handling).to(device)
    # warm start encoder
    sd = torch.load(args.init, map_location='cpu')
    model.encoder.load_state_dict(sd, strict=True)
    print(f"[train] encoder warm-started from {args.init}")
    if args.decoder_ckpt:
        model.decoder.load_state_dict(torch.load(args.decoder_ckpt, map_location='cpu'), strict=True)
        print(f"[train] decoder warm-started from {args.decoder_ckpt}")

    ds = IuxrayPerImage(args.ann_path, args.image_dir, 'train',
                        node_list, node2idx, labeler, build_train_transform())
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.num_workers, drop_last=True)

    mse = nn.MSELoss()
    focal = FocalBCEWithLogits(gamma=args.focal_gamma, alpha=args.focal_alpha)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model.train()
    for epoch in range(1, args.epochs + 1):
        tot = tot_r = tot_c = 0.0
        n = 0
        for images, views, labels in dl:
            images = images.to(device); views = views.to(device); labels = labels.float().to(device)
            recon, logits, _ = model(images, views)
            l_recon = mse(recon, images)
            l_cls = focal(logits, labels)
            loss = args.lambda_recon * l_recon + args.lambda_cls * l_cls
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); tot_r += l_recon.item(); tot_c += l_cls.item(); n += 1
        print(f"[epoch {epoch}/{args.epochs}] loss={tot/n:.4f}  recon={tot_r/n:.4f}  cls={tot_c/n:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    torch.save(model.encoder.state_dict(), args.out)   # drop-in for ae_encoder.pth
    torch.save({'encoder': model.encoder.state_dict(),
                'decoder': model.decoder.state_dict(),
                'head': model.head.state_dict(),
                'view_embed': None if model.view_embed is None else model.view_embed.state_dict(),
                'node_list': node_list, 'args': vars(args)},
               args.out.replace('.pth', '_full.pth'))
    print(f"[train] saved encoder -> {args.out}")


# ----------------------------- Smoke test ------------------------------------
def smoke():
    print("[smoke] combined-loss forward+backward on synthetic tensors ...")
    torch.manual_seed(0)
    n_findings = 74
    B = 4
    model = CombinedAE(n_findings, view_handling='flag').train()
    # NOTE: no ckpt load in smoke — encoder is random init.
    images = torch.randn(B, 3, 224, 224)
    views = torch.randint(0, 2, (B,))
    labels = (torch.rand(B, n_findings) > 0.7).float()

    mse = nn.MSELoss()
    focal = FocalBCEWithLogits()
    recon, logits, z = model(images, views)
    assert recon.shape == (B, 3, 224, 224), recon.shape
    assert logits.shape == (B, n_findings), logits.shape
    assert z.shape == (B, 256, 14, 14), z.shape
    print(f"  recon {tuple(recon.shape)}  logits {tuple(logits.shape)}  z {tuple(z.shape)}")

    l_recon = mse(recon, images)
    l_cls = focal(logits, labels)
    loss = 0.1 * l_recon + 1.0 * l_cls
    assert torch.isfinite(l_recon) and l_recon.item() > 0, l_recon
    assert torch.isfinite(l_cls) and l_cls.item() > 0, l_cls
    print(f"  MSE={l_recon.item():.4f}  FocalBCE={l_cls.item():.4f}  total={loss.item():.4f}")

    model.zero_grad()
    loss.backward()
    g_first = model.encoder.net[0].weight.grad
    g_head = model.head.weight.grad
    assert g_first is not None and g_first.abs().sum() > 0, "encoder first conv got no gradient"
    assert g_head is not None and g_head.abs().sum() > 0, "head got no gradient"
    print(f"  encoder first-conv grad L1 = {float(g_first.abs().sum()):.4f} (non-zero)")
    print(f"  head grad L1               = {float(g_head.abs().sum()):.4f} (non-zero)")

    # view flag reachability: view embedding must receive gradient too
    g_view = model.view_embed.weight.grad
    assert g_view is not None and g_view.abs().sum() > 0, "view embedding got no gradient"
    print(f"  view-embed grad L1         = {float(g_view.abs().sum()):.4f} (non-zero)")

    # focal collapse guard: all-negative logits on all-negative labels -> loss>0 still
    zero_logits = torch.zeros(2, n_findings)
    assert focal(zero_logits, torch.zeros(2, n_findings)).item() > 0
    print("[smoke] PASS (both loss terms finite & non-zero; enc/head/view grads flow)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', default='artifacts/ae_encoder.pth', help='encoder warm-start ckpt')
    ap.add_argument('--decoder_ckpt', default=None)
    ap.add_argument('--out', default='artifacts/ae_encoder_combined.pth')
    ap.add_argument('--lambda_recon', type=float, default=0.1)
    ap.add_argument('--lambda_cls', type=float, default=1.0)
    ap.add_argument('--focal_gamma', type=float, default=2.0)
    ap.add_argument('--focal_alpha', type=float, default=0.25)
    ap.add_argument('--view_handling', default='flag', choices=['flag', 'none'])
    ap.add_argument('--labels', default='negaware', choices=['raw', 'negaware'])
    ap.add_argument('--image_dir', default='data/iu_xray/images/')
    ap.add_argument('--ann_path', default='data/iu_xray/annotation.json')
    ap.add_argument('--kg_co_occur_threshold', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight_decay', type=float, default=5e-5)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if args.smoke:
        smoke()
        return
    train(args)


if __name__ == '__main__':
    main()
