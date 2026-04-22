import torch
import torch.nn as nn


class LanguageModelCriterion(nn.Module):
    def __init__(self, label_smoothing=0.0):
        super(LanguageModelCriterion, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, input, target, mask):
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        nll_loss = -input.gather(2, target.long().unsqueeze(2)).squeeze(2)
        if self.label_smoothing > 0.0:
            smooth_loss = -input.mean(dim=2)
            loss = (1.0 - self.label_smoothing) * nll_loss + self.label_smoothing * smooth_loss
        else:
            loss = nll_loss
        output = (loss * mask).sum() / mask.sum()
        return output


def compute_loss(output, reports_ids, reports_masks, label_smoothing=0.0):
    criterion = LanguageModelCriterion(label_smoothing=label_smoothing)
    loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
    return loss