from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossWeights:
    lambda_gld: float = 1.0
    lambda_geo_action: float = 0.03


@dataclass(frozen=True)
class LossSchedule:
    phase0_warmup_steps: int = 10000
    phase1_geo_ramp_steps: int = 20000
    lambda_gld: float = 1.0
    lambda_geo_action: float = 0.03

    def weights(self, step: int) -> LossWeights:
        if step < self.phase0_warmup_steps:
            return LossWeights(
                lambda_gld=self.lambda_gld,
                lambda_geo_action=0.0,
            )

        ramp_end = self.phase0_warmup_steps + self.phase1_geo_ramp_steps
        if step < ramp_end:
            ratio = float(step - self.phase0_warmup_steps) / float(max(1, self.phase1_geo_ramp_steps))
            return LossWeights(
                lambda_gld=self.lambda_gld,
                lambda_geo_action=self.lambda_geo_action * ratio,
            )

        return LossWeights(
            lambda_gld=self.lambda_gld,
            lambda_geo_action=self.lambda_geo_action,
        )


@dataclass(frozen=True)
class LAMLossInputs:
    f0_hat: torch.Tensor
    f1_hat: torch.Tensor
    f0_target: torch.Tensor
    f1_target: torch.Tensor
    z_tokens: torch.Tensor
    z_vector: torch.Tensor | None = None
    z_geo_tokens_pred: torch.Tensor | None = None
    m_geo_tokens: torch.Tensor | None = None


@dataclass(frozen=True)
class LAMLossOutput:
    total: torch.Tensor
    parts: dict[str, torch.Tensor]


@dataclass(frozen=True)
class PixelLAMLossInputs:
    rgb_hat: torch.Tensor
    rgb_target: torch.Tensor
    z_tokens: torch.Tensor
    z_vector: torch.Tensor | None = None
    f0_hat: torch.Tensor | None = None
    f1_hat: torch.Tensor | None = None
    z_geo_tokens_pred: torch.Tensor | None = None
    m_geo_tokens: torch.Tensor | None = None


def compute_lam_losses(inputs: LAMLossInputs, weights: LossWeights) -> LAMLossOutput:
    f0_target = inputs.f0_target.detach()
    f1_target = inputs.f1_target.detach()
    gld_f0 = F.mse_loss(inputs.f0_hat, f0_target)
    gld_f1 = F.mse_loss(inputs.f1_hat, f1_target)
    gld_f01 = gld_f1 + gld_f0

    unused_output_anchor = torch.zeros((), device=inputs.z_tokens.device, dtype=inputs.z_tokens.dtype)
    unused_output_anchor = unused_output_anchor + inputs.z_tokens.sum() * 0.0
    if inputs.z_vector is not None:
        unused_output_anchor = unused_output_anchor + inputs.z_vector.sum() * 0.0
    if inputs.z_geo_tokens_pred is None:
        geo_token = torch.zeros((), device=inputs.z_tokens.device, dtype=inputs.z_tokens.dtype)
    else:
        geo_token = inputs.z_geo_tokens_pred.sum() * 0.0

    if inputs.z_geo_tokens_pred is not None and inputs.m_geo_tokens is not None:
        geo_token = F.mse_loss(inputs.z_geo_tokens_pred, inputs.m_geo_tokens.detach())
    geo_action = geo_token

    total = (
        weights.lambda_gld * gld_f01
        + weights.lambda_geo_action * geo_action
        + unused_output_anchor
    )
    return LAMLossOutput(
        total=total,
        parts={
            "gld_f0": gld_f0,
            "gld_f1": gld_f1,
            "gld_f01": gld_f01,
            "geo_action": geo_action,
            "geo_token": geo_token,
        },
    )


def compute_pixel_lam_losses(
    inputs: PixelLAMLossInputs,
    lambda_pixel: float = 1.0,
    lambda_geo_action: float = 0.0,
) -> LAMLossOutput:
    if inputs.rgb_hat.shape != inputs.rgb_target.shape:
        raise ValueError(
            f"Predicted and target RGB shapes must match, got "
            f"{tuple(inputs.rgb_hat.shape)} and {tuple(inputs.rgb_target.shape)}"
        )
    if inputs.rgb_hat.ndim != 4 or inputs.rgb_hat.shape[1] != 3:
        raise ValueError(f"Expected RGB tensors [B, 3, H, W], got {tuple(inputs.rgb_hat.shape)}")

    pixel = F.mse_loss(inputs.rgb_hat, inputs.rgb_target.detach())
    # Keep every model output in the graph for DDP configurations that disallow unused parameters.
    output_anchor = inputs.z_tokens.sum() * 0.0
    for optional_output in (inputs.z_vector, inputs.f0_hat, inputs.f1_hat, inputs.z_geo_tokens_pred):
        if optional_output is not None:
            output_anchor = output_anchor + optional_output.sum() * 0.0
    if inputs.z_geo_tokens_pred is not None and inputs.m_geo_tokens is not None:
        geo = F.mse_loss(inputs.z_geo_tokens_pred, inputs.m_geo_tokens.detach())
    else:
        geo = output_anchor * 0.0
    total = float(lambda_pixel) * pixel + float(lambda_geo_action) * geo + output_anchor
    return LAMLossOutput(total=total, parts={"pixel_mse": pixel, "geo_action": geo})
