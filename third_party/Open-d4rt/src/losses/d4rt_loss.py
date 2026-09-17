"""D4RT multi-task loss."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.utils import masked_mean, masked_mean_per_sample


class D4RTLoss(nn.Module):
    def __init__(self, train_cfg: Any) -> None:
        super().__init__()
        self.loss_cfg = train_cfg["loss"]

    def _xyz_preprocess(self, xyz: torch.Tensor, mask: torch.Tensor, normalize_depth: bool, transform_log: bool) -> torch.Tensor:
        out = xyz
        if normalize_depth:
            depth = out[..., 2].abs()
            scale = masked_mean_per_sample(depth, mask).clamp_min(1e-6).unsqueeze(-1)
            out = out / scale
        if transform_log:
            out = torch.sign(out) * torch.log1p(out.abs())
        return out

    def _compute_xyz_loss(
        self,
        outputs: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        mask: dict[str, torch.Tensor],
        metrics: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        cfg = self.loss_cfg["xyz_3d"]
        if not bool(cfg.get("enabled", True)):
            return outputs["xyz_3d"].new_zeros(())

        m = mask["xyz_3d"]
        pred = outputs["xyz_3d"]
        gt = target["xyz_3d"]
        use_norm = bool(cfg.get("normalize_by_mean_depth", False))
        use_log = str(cfg.get("value_transform", "")) == "sign_x_log1p_abs_x"
        pred = self._xyz_preprocess(pred, m, use_norm, use_log)
        gt = self._xyz_preprocess(gt, m, use_norm, use_log)

        xyz_err_raw = (pred - gt).abs().mean(dim=-1)
        xyz_err = xyz_err_raw

        conf_cfg = self.loss_cfg.get("confidence", {})
        conf_total = pred.new_zeros(())
        if bool(conf_cfg.get("enabled", False)):
            if "confidence" not in outputs:
                raise RuntimeError("loss.confidence.enabled=true but model output has no 'confidence' head")
            c = torch.sigmoid(outputs["confidence"]).clamp(1e-4, 1.0 - 1e-4)
            if bool(conf_cfg.get("confidence_weights_xyz_error", False)):
                xyz_err = xyz_err * c

            conf_penalty = masked_mean(-torch.log(c), m) * float(conf_cfg.get("weight_lambda_conf", 0.0))
            conf_total = conf_total + conf_penalty
            metrics["loss_confidence_penalty"] = conf_penalty
            metrics["confidence_mean"] = masked_mean(c, m)

            mode = str(conf_cfg.get("mode", "main_text")).strip().lower()
            lconf_cfg = conf_cfg.get("lconf_ablation", {})
            lconf_enabled = bool(lconf_cfg.get("enabled", False)) or mode == "appendix_lconf"
            if mode not in {"main_text", "appendix_lconf"}:
                raise ValueError(f"Unsupported loss.confidence.mode: {mode}")
            if lconf_enabled:
                lconf_type = str(lconf_cfg.get("type", "paper_unspecified")).strip().lower()
                if lconf_type in {"paper_unspecified", "none", ""}:
                    raise RuntimeError(
                        "loss.confidence appendix_lconf is enabled, but `lconf_ablation.type` is not explicitly set. "
                        "Paper appendix introduces L_conf term but does not clearly specify a unique formula in extracted text. "
                        "Set an explicit ablation type to avoid silent assumption."
                    )
                if lconf_type != "exp_neg_abs_xyz_error_l1":
                    raise ValueError(f"Unsupported loss.confidence.lconf_ablation.type: {lconf_type}")
                target_conf = torch.exp(-xyz_err_raw.detach()).clamp(1e-4, 1.0 - 1e-4)
                lconf_raw = (c - target_conf).abs()
                lconf_weight = float(lconf_cfg.get("weight_lambda_lconf", conf_cfg.get("weight_lambda_conf", 0.0)))
                lconf_loss = masked_mean(lconf_raw, m) * lconf_weight
                conf_total = conf_total + lconf_loss
                metrics["loss_confidence_lconf"] = lconf_loss
                metrics["confidence_target_mean"] = masked_mean(target_conf, m)

        xyz_loss = masked_mean(xyz_err, m) * float(cfg.get("weight_lambda_3d", 1.0))
        metrics["loss_xyz_3d"] = xyz_loss
        return xyz_loss + conf_total

    def _compute_uv_loss(
        self,
        outputs: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        mask: dict[str, torch.Tensor],
        metrics: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        cfg = self.loss_cfg["uv_2d"]
        if not bool(cfg.get("enabled", True)):
            return outputs["uv_2d"].new_zeros(())
        uv_err = (outputs["uv_2d"] - target["uv_2d"]).abs().mean(dim=-1)
        uv_loss = masked_mean(uv_err, mask["uv_2d"]) * float(cfg.get("weight_lambda_2d", 0.1))
        metrics["loss_uv_2d"] = uv_loss
        return uv_loss

    def _project_pred_xyz_to_tgt_uv(
        self,
        *,
        pred_xyz_tcam: torch.Tensor,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        camera = batch.get("camera", {})
        query = batch.get("query", {})
        video = batch.get("video")
        if not isinstance(camera, dict) or not isinstance(query, dict) or (not torch.is_tensor(video)):
            return None, None
        k_seq = camera.get("K")
        t_wc_seq = camera.get("T_wc")
        cam_valid_seq = camera.get("camera_valid")
        t_cam = query.get("t_cam")
        t_tgt = query.get("t_tgt")
        if not all(torch.is_tensor(x) for x in (k_seq, t_wc_seq, cam_valid_seq, t_cam, t_tgt)):
            return None, None

        bsz, _, _, image_h, image_w = video.shape
        if pred_xyz_tcam.ndim != 3 or pred_xyz_tcam.shape[0] != bsz:
            return None, None
        num_queries = pred_xyz_tcam.shape[1]
        batch_idx = torch.arange(bsz, device=pred_xyz_tcam.device)[:, None]
        t_cam_idx = t_cam.long().clamp_min(0)
        t_tgt_idx = t_tgt.long().clamp_min(0)

        k_tgt = k_seq[batch_idx, t_tgt_idx]
        t_wc_cam = t_wc_seq[batch_idx, t_cam_idx]
        t_wc_tgt = t_wc_seq[batch_idx, t_tgt_idx]
        cam_valid = cam_valid_seq[batch_idx, t_cam_idx] & cam_valid_seq[batch_idx, t_tgt_idx]

        ones = torch.ones((*pred_xyz_tcam.shape[:2], 1), dtype=pred_xyz_tcam.dtype, device=pred_xyz_tcam.device)
        pred_xyz_h = torch.cat([pred_xyz_tcam, ones], dim=-1)
        world_h = torch.matmul(t_wc_cam, pred_xyz_h.unsqueeze(-1)).squeeze(-1)
        try:
            t_cw_tgt = torch.linalg.inv(t_wc_tgt)
        except RuntimeError:
            return None, None
        pred_tgt = torch.matmul(t_cw_tgt, world_h.unsqueeze(-1)).squeeze(-1)[..., :3]
        z = pred_tgt[..., 2]
        proj = torch.matmul(k_tgt, pred_tgt.unsqueeze(-1)).squeeze(-1)
        safe_z = torch.where(z.abs() > 1e-6, z, torch.ones_like(z))
        u_px = proj[..., 0] / safe_z
        v_px = proj[..., 1] / safe_z
        uv_proj = torch.stack(
            [
                u_px / float(max(int(image_w) - 1, 1)),
                v_px / float(max(int(image_h) - 1, 1)),
            ],
            dim=-1,
        )
        proj_valid = cam_valid & torch.isfinite(pred_tgt).all(dim=-1) & torch.isfinite(uv_proj).all(dim=-1) & (z > 1e-6)
        return uv_proj, proj_valid

    def _compute_reprojection_uv_from_xyz_loss(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, Any],
        metrics: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        cfg = self.loss_cfg.get("reprojection_uv_from_xyz", {})
        if not bool(cfg.get("enabled", False)):
            return outputs["xyz_3d"].new_zeros(())

        target = batch["target"]
        mask = batch["mask"]
        uv_proj, proj_valid = self._project_pred_xyz_to_tgt_uv(pred_xyz_tcam=outputs["xyz_3d"], batch=batch)
        if uv_proj is None or proj_valid is None:
            metrics["loss_reprojection_uv_from_xyz"] = outputs["xyz_3d"].new_zeros(())
            return outputs["xyz_3d"].new_zeros(())

        reproj_mask = mask["xyz_3d"] & mask["uv_2d"] & proj_valid
        if bool(cfg.get("detach_xyz", False)):
            uv_proj = uv_proj.detach()

        image_h = int(batch["video"].shape[-2])
        image_w = int(batch["video"].shape[-1])
        scale_xy = uv_proj.new_tensor(
            [
                float(max(image_w - 1, 1)),
                float(max(image_h - 1, 1)),
            ]
        )
        diff_px = (uv_proj - target["uv_2d"]) * scale_xy
        diff_px = torch.where(reproj_mask.unsqueeze(-1), diff_px, torch.zeros_like(diff_px))

        robust_loss = str(cfg.get("robust_loss", "huber")).strip().lower()
        if robust_loss in {"huber", "smooth_l1"}:
            huber_delta_px = float(cfg.get("huber_delta_px", 8.0))
            reproj_err = F.huber_loss(
                diff_px,
                torch.zeros_like(diff_px),
                reduction="none",
                delta=huber_delta_px,
            ).mean(dim=-1)
        elif robust_loss in {"l1", "mae"}:
            reproj_err = diff_px.abs().mean(dim=-1)
        else:
            raise ValueError(f"Unsupported loss.reprojection_uv_from_xyz.robust_loss: {robust_loss}")

        reproj_err = torch.where(reproj_mask, reproj_err, torch.zeros_like(reproj_err))
        reproj_loss = masked_mean(reproj_err, reproj_mask) * float(cfg.get("weight_lambda_reproj_uv", 0.05))
        metrics["loss_reprojection_uv_from_xyz"] = reproj_loss

        reproj_abs_err_px = torch.where(reproj_mask, diff_px.abs().mean(dim=-1), torch.zeros_like(reproj_err))
        metrics["reprojection_uv_from_xyz_error_mean_px"] = masked_mean(reproj_abs_err_px, reproj_mask)
        return reproj_loss

    def _compute_vis_loss(
        self,
        outputs: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        mask: dict[str, torch.Tensor],
        metrics: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        cfg = self.loss_cfg["visibility"]
        if not bool(cfg.get("enabled", True)):
            return outputs["visibility"].new_zeros(())
        bce = F.binary_cross_entropy_with_logits(outputs["visibility"], target["visibility"], reduction="none")
        vis_loss = masked_mean(bce, mask["visibility"]) * float(cfg.get("weight_lambda_vis", 0.1))
        metrics["loss_visibility"] = vis_loss
        return vis_loss

    def _compute_disp_loss(
        self,
        outputs: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        mask: dict[str, torch.Tensor],
        metrics: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        cfg = self.loss_cfg["displacement"]
        if not bool(cfg.get("enabled", False)) or "displacement" not in outputs:
            return outputs["xyz_3d"].new_zeros(())
        disp_err = (outputs["displacement"] - target["displacement"]).abs().mean(dim=-1)
        disp_loss = masked_mean(disp_err, mask["displacement"]) * float(cfg.get("weight_lambda_disp", 0.1))
        metrics["loss_displacement"] = disp_loss
        return disp_loss

    def _compute_normal_loss(
        self,
        outputs: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        mask: dict[str, torch.Tensor],
        metrics: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        cfg = self.loss_cfg["normal"]
        if not bool(cfg.get("enabled", False)) or "normal" not in outputs:
            return outputs["xyz_3d"].new_zeros(())
        pred = F.normalize(outputs["normal"], dim=-1)
        gt = F.normalize(target["normal"], dim=-1)
        cos = F.cosine_similarity(pred, gt, dim=-1)
        norm_loss = masked_mean(1.0 - cos, mask["normal"]) * float(cfg.get("weight_lambda_normal", 0.5))
        metrics["loss_normal"] = norm_loss
        return norm_loss

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target = batch["target"]
        mask = batch["mask"]
        metrics: dict[str, torch.Tensor] = {}

        total = outputs["xyz_3d"].new_zeros(())
        total = total + self._compute_xyz_loss(outputs, target, mask, metrics)
        total = total + self._compute_uv_loss(outputs, target, mask, metrics)
        total = total + self._compute_reprojection_uv_from_xyz_loss(outputs, batch, metrics)
        total = total + self._compute_vis_loss(outputs, target, mask, metrics)
        total = total + self._compute_disp_loss(outputs, target, mask, metrics)
        total = total + self._compute_normal_loss(outputs, target, mask, metrics)

        metrics["loss_total"] = total
        return total, metrics
