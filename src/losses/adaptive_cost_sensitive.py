import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveCostSensitiveLoss(nn.Module):
    """
    Adaptive cost-sensitive loss:

    - Base loss: BCEWithLogits per sample
    - Class cost: higher weight for positive (fraud) samples
    - Hardness: samples with consistently high loss get extra weight

    loss = mean( weight_i * bce_i )

    where weight_i = class_cost_i * (1 + hardness_scale * hardness_norm_i)
    """
    def __init__(
        self,
        pos_cost: float = 10.0,
        neg_cost: float = 1.0,
        hardness_scale: float = 0.5,

    ):
        super().__init__()
        self.pos_cost = float(pos_cost)
        self.neg_cost = float(neg_cost)
        self.hardness_scale = float(hardness_scale)

    def forward(self, logits, targets, sample_ids, hardness=None):
        """
        Args:
            logits: Tensor [B] or [B, 1]
            targets: Tensor [B] (float 0/1)
            sample_ids: Tensor [B] (indices into global hardness buffer)
            hardness: Tensor [N] or None (global hardness values, optional)

        Returns:
            loss: scalar tensor
            base_loss_detached: Tensor [B] (per-sample BCE loss, detached)
        """
        # ensure shapes
        if logits.ndim > 1:
            logits = logits.squeeze(-1)

        base_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )  # [B]

        # class cost: positives use pos_cost, negatives use neg_cost
        class_costs = torch.where(
            targets > 0.5,
            torch.tensor(self.pos_cost, dtype=base_loss.dtype, device=base_loss.device),
            torch.tensor(self.neg_cost, dtype=base_loss.dtype, device=base_loss.device),
        )  # [B]

        weights = class_costs  # start with pure cost-based weight

        # if hardness buffer is provided, use it
        if hardness is not None:
            # hardness indexed by sample_ids
            h_vals = hardness[sample_ids]  # [B]

            # normalize hardness to [0, 1] to avoid exploding weights
            h_min = torch.min(h_vals)
            h_max = torch.max(h_vals)
            denom = (h_max - h_min).clamp_min(1e-8)
            h_norm = (h_vals - h_min) / denom  # [0,1]

            # adaptive factor: (1 + hardness_scale * h_norm)
            hardness_factor = 1.0 + self.hardness_scale * h_norm
            weights = weights * hardness_factor

        # compute weighted loss
        loss = (weights * base_loss).mean()

        return loss, base_loss.detach()
