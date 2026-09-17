from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .config import load_lam_config
from .integrations.gld import GLDBackboneWrapper
from .models.gld_lam import GLDLAM


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transfer a LAM latent action to a new frame and decode predicted RGB")
    parser.add_argument("--checkpoint", type=Path, required=True, help="LAM checkpoint, e.g. outputs/taco_lam/lam_last.pt")
    parser.add_argument("--action-src", type=Path, required=True, help="Image I_t used to infer the latent action")
    parser.add_argument("--action-tgt", type=Path, required=True, help="Image I_t+k used to infer the latent action")
    parser.add_argument("--condition", type=Path, required=True, help="New frame I'_t to which the latent action is applied")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/lam_3d_pixel.yaml"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gld-mae-weight", type=Path, default=Path("third_party/GLD/pretrained_models/mae_decoder.pt"))
    parser.add_argument("--decoder-config", type=Path, default=Path("configs/gld_mae_decoder"))
    return parser.parse_args(argv)


def tokens_to_feature_map(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"Expected patch tokens [B, N, C], got {tuple(tokens.shape)}")
    batch, num_tokens, channels = tokens.shape
    side = int(num_tokens**0.5)
    if side * side != num_tokens:
        raise ValueError(f"Token count must be square for 2D feature map conversion, got {num_tokens}")
    return tokens.transpose(1, 2).reshape(batch, channels, side, side).contiguous()


def _raw_level1_for_propagation(
    gld: GLDBackboneWrapper,
    raw_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare raw L1 patches/CLS for GLD's latent-normalized propagation API."""
    feature_map = tokens_to_feature_map(raw_tokens)
    raw_cls = torch.zeros(
        raw_tokens.shape[0], raw_tokens.shape[-1], device=raw_tokens.device, dtype=raw_tokens.dtype
    )
    if not getattr(gld.model, "do_normalization", False):
        return feature_map, raw_cls
    normalized = gld.model._normalize(feature_map, raw_cls)
    if not isinstance(normalized, tuple) or len(normalized) != 2:
        raise RuntimeError("GLD normalization did not return normalized patch and CLS features")
    return normalized


def decode_predicted_rgb(
    gld: GLDBackboneWrapper,
    f0_hat: torch.Tensor,
    f1_hat: torch.Tensor,
    height: int,
    width: int,
    *,
    denormalize: bool = True,
) -> torch.Tensor:
    if getattr(gld.model, "mae_decoder", None) is None:
        raise RuntimeError("GLD MAE decoder is not loaded. Pass --gld-mae-weight to enable RGB decoding.")

    if f0_hat.shape != f1_hat.shape:
        raise ValueError(f"F0/F1 shapes must match, got {tuple(f0_hat.shape)} and {tuple(f1_hat.shape)}")

    f1_for_propagation, cls_for_propagation = _raw_level1_for_propagation(gld, f1_hat)
    propagated = gld.model.propagate_features(
        f1_for_propagation,
        from_level=1,
        total_view=1,
        cls_token=cls_for_propagation,
        normalize_patches=False,
    )
    if len(propagated) < 3:
        raise RuntimeError(f"Expected propagated levels 1-3, got {len(propagated)} levels")

    dummy_cls = torch.zeros(
        f0_hat.shape[0], 1, f0_hat.shape[-1], device=f0_hat.device, dtype=f0_hat.dtype
    )
    feats = [
        # The shipped MAE checkpoint was trained on raw mode="all" features.
        (f0_hat.unsqueeze(1), dummy_cls),
        propagated[0],
        propagated[1],
        propagated[2],
    ]
    rgb = gld.model.decode(feats, H=height, W=width)["rgb"]
    if rgb.ndim == 5:
        rgb = rgb[:, 0]
    if not denormalize:
        mean = gld.model.encoder_mean.to(device=rgb.device, dtype=rgb.dtype)
        std = gld.model.encoder_std.to(device=rgb.device, dtype=rgb.dtype)
        rgb = (rgb - mean) / std
    return rgb.clamp(0.0, 1.0)


def decode_training_consistent_rgb(
    gld: GLDBackboneWrapper,
    f0_hat: torch.Tensor,
    f1_hat: torch.Tensor,
    image_size: int,
) -> torch.Tensor:
    """Decode raw predicted GLD features for RGB visualization."""
    return decode_predicted_rgb(gld, f0_hat, f1_hat, image_size, image_size)


def load_image_tensor(path: Path, image_size: int, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((image_size, image_size), Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor.to(device)


def save_image_tensor(tensor: torch.Tensor, path: Path) -> None:
    image = tensor.detach().cpu().clamp(0.0, 1.0)
    if image.ndim == 4:
        image = image[0]
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(array).save(path)


def make_labeled_collage(images: list[torch.Tensor], labels: list[str], path: Path) -> None:
    """Save a single horizontal, labeled comparison image."""
    if len(images) != len(labels) or not images:
        raise ValueError("images and labels must be non-empty and have the same length")
    pil_images = []
    for tensor in images:
        image = tensor.detach().cpu().clamp(0.0, 1.0)
        if image.ndim == 4:
            image = image[0]
        array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        pil_images.append(Image.fromarray(array).convert("RGB"))
    width, height = pil_images[0].size
    if any(image.size != (width, height) for image in pil_images):
        raise ValueError("All collage images must have the same dimensions")
    gap = 12
    header = 42
    canvas = Image.new("RGB", (len(pil_images) * width + (len(pil_images) - 1) * gap, height + header), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    for index, (image, label) in enumerate(zip(pil_images, labels)):
        x = index * (width + gap)
        draw.text((x + width // 2, 8), label, fill="black", font=font, anchor="ma")
        canvas.paste(image, (x, header))
    canvas.save(path)


def build_lam_model(cfg) -> GLDLAM:
    return GLDLAM(
        in_dim=cfg.model.f_dim,
        c_state=cfg.model.c_state,
        c_z=cfg.model.c_z,
        z_dim=cfg.model.z_dim,
        c_geo=cfg.model.c_geo,
        k_action=cfg.model.k_action,
        k_geo=cfg.model.k_geo,
        latent_dim=cfg.model.latent_dim,
        heads=cfg.model.heads,
        action_bottleneck_depth=cfg.model.action_bottleneck_depth,
        idm_depth=cfg.model.idm_depth,
        fdm_depth=cfg.model.fdm_depth,
        fdm_self_attn_every=cfg.model.fdm_self_attn_every,
        ffn_ratio=cfg.model.ffn_ratio,
    )


def main() -> int:
    args = parse_args()
    cfg = load_lam_config(args.config)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gld = GLDBackboneWrapper(
        cfg.paths.gld_root,
        checkpoint=cfg.paths.gld_checkpoint,
        encoder_pretrained_path=cfg.paths.gld_encoder_pretrained_path,
        mae_weight=args.gld_mae_weight,
        decoder_config_path=args.decoder_config,
        image_size=cfg.data.image_size,
    ).to(device)
    lam = build_lam_model(cfg).to(device)
    payload = torch.load(args.checkpoint, map_location="cpu")
    state_dict = payload.get("model", payload)
    result = lam.load_state_dict(state_dict, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    if missing or unexpected:
        warnings.warn(
            f"LAM checkpoint loaded with missing_keys={missing} unexpected_keys={unexpected}",
            RuntimeWarning,
            stacklevel=2,
        )
    lam.eval()

    action_src = load_image_tensor(args.action_src, cfg.data.image_size, device)
    action_tgt = load_image_tensor(args.action_tgt, cfg.data.image_size, device)
    condition = load_image_tensor(args.condition, cfg.data.image_size, device)

    with torch.no_grad():
        f0_src, f1_src = gld(action_src)
        f0_tgt, f1_tgt = gld(action_tgt)
        action_out = lam(f0_src, f1_src, f0_tgt, f1_tgt)

        f0_cond, f1_cond = gld(condition)
        state_cond = lam.state_encoder(f0_cond, f1_cond)
        pred_tokens = lam.fdm(state_cond.tokens, action_out.latent_action.tokens)
        f0_hat, f1_hat = lam.feature_head(pred_tokens, f0_cond, f1_cond)
        pred_rgb = decode_training_consistent_rgb(gld, f0_hat, f1_hat, cfg.data.image_size)

    pred_rgb_path = args.output_dir / "pred_rgb.png"
    save_image_tensor(pred_rgb, pred_rgb_path)

    collage_path = args.output_dir / "prediction_comparison.png"
    make_labeled_collage(
        [action_src, action_tgt, condition, pred_rgb],
        ["SRC", "TGT", "CONDITION", "PREDICTED"],
        collage_path,
    )
    torch.save(
        {
            "z_tokens": action_out.latent_action.tokens.detach().cpu(),
            "z_vec": action_out.latent_action.vector.detach().cpu(),
            "z_low_dim": action_out.latent_action.low_dim.detach().cpu(),
            "latent_type": "deterministic_normalized",
            "f0_hat": f0_hat.detach().cpu(),
            "f1_hat": f1_hat.detach().cpu(),
            "checkpoint": str(args.checkpoint),
            "action_src": str(args.action_src),
            "action_tgt": str(args.action_tgt),
            "condition": str(args.condition),
        },
        args.output_dir / "latent_action.pt",
    )
    metadata = {
        "latent_type": "deterministic_normalized",
        "checkpoint": str(args.checkpoint),
        "action_src": str(args.action_src),
        "action_tgt": str(args.action_tgt),
        "condition": str(args.condition),
        "pred_rgb": str(pred_rgb_path),
        "prediction_comparison": str(collage_path),
        "pred_rgb_preprocessing": "raw_gld_f0_plus_raw_f1_f3_propagation",
        "latent_action": str(args.output_dir / "latent_action.pt"),
        "note": "The transferred latent action comes from action_src/action_tgt; prediction is action-conditioned, not an unconditional one-frame future prior.",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved_pred_rgb={pred_rgb_path}", flush=True)
    print(f"saved_prediction_comparison={collage_path}", flush=True)
    print(f"saved_latent_action={args.output_dir / 'latent_action.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
