from __future__ import annotations

import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class D4RTPairOutput:
    hidden_t: torch.Tensor
    heads_t: dict[str, torch.Tensor]
    hidden_tk: torch.Tensor
    heads_tk: dict[str, torch.Tensor]


def _expand_query_time(
    value: int | torch.Tensor,
    batch_size: int,
    num_queries: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """Normalize scalar or per-batch query times to [B, M] long tensors."""
    tensor = torch.as_tensor(value, device=device, dtype=torch.long)
    if tensor.ndim == 0:
        tensor = tensor.view(1).expand(batch_size)
    if tensor.ndim == 1:
        if tensor.shape[0] == 1 and batch_size != 1:
            tensor = tensor.expand(batch_size)
        if tensor.shape[0] != batch_size:
            raise ValueError(f"{name} must be scalar, [B], or [B,M], got {tuple(tensor.shape)} for B={batch_size}")
        tensor = tensor.view(batch_size, 1).expand(batch_size, num_queries)
    elif tensor.ndim == 2:
        if tensor.shape != (batch_size, num_queries):
            raise ValueError(
                f"{name} must be scalar, [B], or [B,M], got {tuple(tensor.shape)} "
                f"for expected [B={batch_size}, M={num_queries}]"
            )
    else:
        raise ValueError(f"{name} must be scalar, [B], or [B,M], got {tuple(tensor.shape)}")
    return tensor.contiguous()


def build_d4rt_clip_queries(
    batch_size: int,
    grid_size: int = 32,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    t_src: int | torch.Tensor = 0,
    t_tgt: int | torch.Tensor = 1,
    t_cam: int | torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Build current/future queries for a clip with explicit local times."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    device = torch.device("cpu") if device is None else torch.device(device)
    coords = torch.linspace(0.0, 1.0, grid_size, device=device, dtype=dtype)
    v_grid, u_grid = torch.meshgrid(coords, coords, indexing="ij")
    num_queries = grid_size * grid_size
    u = u_grid.reshape(1, -1).expand(batch_size, -1).contiguous()
    v = v_grid.reshape(1, -1).expand(batch_size, -1).contiguous()
    src = _expand_query_time(t_src, batch_size, num_queries, device, "t_src")
    tgt = _expand_query_time(t_tgt, batch_size, num_queries, device, "t_tgt")
    cam = _expand_query_time(t_src if t_cam is None else t_cam, batch_size, num_queries, device, "t_cam")

    q_cur = {"u": u, "v": v, "t_src": src, "t_tgt": src.clone(), "t_cam": cam}
    q_fut = {"u": u.clone(), "v": v.clone(), "t_src": src.clone(), "t_tgt": tgt, "t_cam": cam.clone()}
    return q_cur, q_fut


def build_d4rt_pair_queries(
    batch_size: int,
    grid_size: int = 32,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    return build_d4rt_clip_queries(
        batch_size=batch_size,
        grid_size=grid_size,
        device=device,
        dtype=dtype,
        t_src=0,
        t_tgt=1,
        t_cam=0,
    )


@contextmanager
def _prepend_sys_path(path: Path):
    path_str = str(path)
    inserted = False
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(path_str)
            except ValueError:
                pass


def _unwrap_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("model", "state_dict", "module", "network", "net"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        if payload and all(torch.is_tensor(v) for v in payload.values()):
            return payload
    return {}


class D4RTWrapper(torch.nn.Module):
    """Runtime wrapper for OpenD4RT teacher extraction."""

    def __init__(
        self,
        d4rt_root: str | Path,
        model_config: str | Path,
        checkpoint: str | Path,
        query_grid_size: int = 32,
        image_size: int = 256,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.d4rt_root = Path(d4rt_root)
        self.model_config = Path(model_config)
        self.checkpoint = Path(checkpoint)
        self.query_grid_size = int(query_grid_size)
        self.image_size = int(image_size)
        self.runtime_device = torch.device("cpu") if device is None else torch.device(device)
        self.model = self._load_model()
        self.model.to(self.runtime_device).eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def _load_model(self) -> torch.nn.Module:
        with _prepend_sys_path(self.d4rt_root):
            from src.core.config import load_yaml_config
            from src.model.builder import build_model

            cfg = load_yaml_config(self.model_config)
            model = build_model(cfg["model"]).eval()

        payload = torch.load(self.checkpoint, map_location="cpu")
        state_dict = _unwrap_state_dict(payload)
        if not state_dict:
            raise RuntimeError(f"No model weights found in D4RT checkpoint: {self.checkpoint}")
        result = model.load_state_dict(state_dict, strict=False)
        missing = list(result.missing_keys)
        unexpected = list(result.unexpected_keys)
        if missing or unexpected:
            warnings.warn(
                f"D4RT checkpoint loaded with missing_keys={missing} unexpected_keys={unexpected}",
                RuntimeWarning,
                stacklevel=2,
            )
        return model

    def _prepare_video(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(f"Expected video tensor shaped [B, T, C, H, W], got {video.shape}")
        if video.shape[1] < 2:
            raise ValueError("D4RT clips must contain at least two frames")
        video = video.to(self.runtime_device)
        if not video.is_floating_point():
            video = video.float()
        b, t, c, h, w = video.shape
        if h != self.image_size or w != self.image_size:
            video = video.reshape(b * t, c, h, w)
            video = F.interpolate(video, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
            video = video.reshape(b, t, c, self.image_size, self.image_size)
        return video

    @staticmethod
    def _normalize_times(
        value: int | torch.Tensor,
        batch_size: int,
        clip_frames: int,
        name: str,
        device: torch.device,
    ) -> torch.Tensor:
        times = torch.as_tensor(value, device=device, dtype=torch.long)
        if times.ndim == 0:
            times = times.view(1).expand(batch_size)
        elif times.ndim == 1:
            if times.shape[0] == 1 and batch_size != 1:
                times = times.expand(batch_size)
            if times.shape[0] != batch_size:
                raise ValueError(f"{name} must have one value per batch item, got {tuple(times.shape)}")
        else:
            raise ValueError(f"{name} must be scalar or [B], got {tuple(times.shape)}")
        if torch.any(times < 0) or torch.any(times >= clip_frames):
            raise ValueError(f"{name} values must be in [0, {clip_frames}), got {times.tolist()}")
        return times.contiguous()

    def _decode_with_hidden(
        self,
        video: torch.Tensor,
        query: dict[str, torch.Tensor],
        memory: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        u = query["u"].to(device=video.device, dtype=video.dtype)
        v = query["v"].to(device=video.device, dtype=video.dtype)
        t_src = query["t_src"].to(device=video.device).long()
        t_tgt = query["t_tgt"].to(device=video.device).long()
        t_cam = query["t_cam"].to(device=video.device).long()

        query_tokens = self.model.query_embedder(video=video, u=u, v=v, t_src=t_src, t_tgt=t_tgt, t_cam=t_cam)
        hidden = self.model.decoder(query_tokens, memory)
        return hidden, self.model.heads(hidden)

    @torch.no_grad()
    def forward_clip(
        self,
        video: torch.Tensor,
        t_src: int | torch.Tensor = 0,
        t_tgt: int | torch.Tensor = 1,
        t_cam: int | torch.Tensor | None = None,
    ) -> D4RTPairOutput:
        video = self._prepare_video(video)
        clip_frames = int(video.shape[1])
        query_embedder = getattr(self.model, "query_embedder", None)
        max_frames = int(getattr(query_embedder, "max_frames", clip_frames))
        if clip_frames > max_frames:
            raise ValueError(f"D4RT clip has {clip_frames} frames but the checkpoint supports at most {max_frames}")
        src = self._normalize_times(t_src, video.shape[0], clip_frames, "t_src", video.device)
        tgt = self._normalize_times(t_tgt, video.shape[0], clip_frames, "t_tgt", video.device)
        cam = self._normalize_times(src if t_cam is None else t_cam, video.shape[0], clip_frames, "t_cam", video.device)
        if torch.any(tgt < src):
            raise ValueError("t_tgt must be greater than or equal to t_src")
        q_cur, q_fut = build_d4rt_clip_queries(
            batch_size=video.shape[0],
            grid_size=self.query_grid_size,
            device=video.device,
            dtype=video.dtype,
            t_src=src,
            t_tgt=tgt,
            t_cam=cam,
        )
        memory = self.model.encode_video(video=video)
        hidden_t, heads_t = self._decode_with_hidden(video, q_cur, memory)
        hidden_tk, heads_tk = self._decode_with_hidden(video, q_fut, memory)
        return D4RTPairOutput(
            hidden_t=hidden_t.detach(),
            heads_t={key: value.detach() for key, value in heads_t.items()},
            hidden_tk=hidden_tk.detach(),
            heads_tk={key: value.detach() for key, value in heads_tk.items()},
        )

    @torch.no_grad()
    def forward_pair(self, image_t: torch.Tensor, image_tk: torch.Tensor) -> D4RTPairOutput:
        if image_t.ndim != 4 or image_tk.ndim != 4:
            raise ValueError("Expected image tensors shaped [B, C, H, W]")
        if image_t.shape != image_tk.shape:
            raise ValueError(f"Pair image shapes must match, got {image_t.shape} and {image_tk.shape}")
        video = torch.stack([image_t, image_tk], dim=1)
        return self.forward_clip(video, t_src=0, t_tgt=1, t_cam=0)
