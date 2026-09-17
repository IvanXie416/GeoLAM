"""Shared query-supervision builder from dense depth + camera trajectories."""

from __future__ import annotations

import cv2
import numpy as np

from .raw_augment import (
    depth_boundary_mask,
    sample_hard_query_flags,
    sample_t_tgt_t_cam,
)


def _compute_normal_map(
    depth: np.ndarray,
    k: np.ndarray,
    depth_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pseudo surface normals from a depth map via cross-product.

    Returns (normal_hw3, normal_valid_hw).
    Normals are in the camera coordinate frame, pointing toward the camera
    (i.e. z-component is typically negative for surfaces facing the camera).

    The depth map is Gaussian-smoothed before differentiation to suppress
    noise-induced normal errors (raw ARKit/sensor depth has ~1-3cm noise;
    without smoothing, a 1-pixel central difference can produce 40-70° errors).
    """
    h, w = depth.shape
    if h < 3 or w < 3:
        return np.zeros((h, w, 3), dtype=np.float32), np.zeros((h, w), dtype=bool)

    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    if abs(fx) < 1e-6 or abs(fy) < 1e-6:
        return np.zeros((h, w, 3), dtype=np.float32), np.zeros((h, w), dtype=bool)

    # Smooth depth to suppress sensor noise before differentiation.
    # Fill invalid pixels with 0 for the blur, then restore the valid mask.
    z_raw = depth.astype(np.float32)
    z_filled = np.where(depth_valid, z_raw, 0.0).astype(np.float32)
    weight = depth_valid.astype(np.float32)
    ksize = 7
    sigma = 2.0
    z_blur = cv2.GaussianBlur(z_filled, (ksize, ksize), sigma)
    w_blur = cv2.GaussianBlur(weight, (ksize, ksize), sigma)
    # Weighted average: only count valid neighbors.
    z = np.zeros_like(z_blur, dtype=np.float32)
    np.divide(z_blur, w_blur, out=z, where=(w_blur > 1e-6))

    # Unproject smoothed depth to 3D point cloud in camera frame.
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy

    # Horizontal and vertical finite differences (central difference).
    # dx = P[v, u+1] - P[v, u-1],  dy = P[v+1, u] - P[v-1, u]
    dx = np.zeros((h, w, 3), dtype=np.float32)
    dy = np.zeros((h, w, 3), dtype=np.float32)
    dx[1:-1, 1:-1, 0] = x[1:-1, 2:] - x[1:-1, :-2]
    dx[1:-1, 1:-1, 1] = y[1:-1, 2:] - y[1:-1, :-2]
    dx[1:-1, 1:-1, 2] = z[1:-1, 2:] - z[1:-1, :-2]
    dy[1:-1, 1:-1, 0] = x[2:, 1:-1] - x[:-2, 1:-1]
    dy[1:-1, 1:-1, 1] = y[2:, 1:-1] - y[:-2, 1:-1]
    dy[1:-1, 1:-1, 2] = z[2:, 1:-1] - z[:-2, 1:-1]

    # cross(dx, dy) points away from camera; negate for toward-camera convention.
    raw = np.cross(dx, dy)
    norm = np.linalg.norm(raw, axis=-1, keepdims=True)
    safe_norm = np.maximum(norm, 1e-8)
    normal = np.where(norm > 1e-8, -raw / safe_norm, 0.0).astype(np.float32)

    # Validity: interior pixels with sufficient valid neighbor coverage.
    valid = np.zeros((h, w), dtype=bool)
    valid[1:-1, 1:-1] = (
        depth_valid[1:-1, 1:-1]
        & depth_valid[1:-1, :-2]
        & depth_valid[1:-1, 2:]
        & depth_valid[:-2, 1:-1]
        & depth_valid[2:, 1:-1]
    )
    return normal, valid


def build_queries_from_depth(
    rng: np.random.Generator,
    depth: np.ndarray,
    depth_valid: np.ndarray,
    k_seq: np.ndarray,
    t_wc_seq: np.ndarray,
    camera_valid: np.ndarray,
    queries_per_clip: int,
    hard_query_ratio: float,
    prob_t_tgt_equals_t_cam: float,
    t_src_tgt_delta_choices: tuple[int | None, ...] | None = None,
    t_src_tgt_delta_probs: tuple[float, ...] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build D4RT query/target/mask/query_stats from depth and camera.

    This assumes static-scene geometry for correspondence construction:
    a source-depth point is unprojected to world and then reprojected to target/cam frames.
    """

    t, h, w = depth.shape
    m = int(queries_per_clip)

    q_t_src = rng.integers(0, t, size=(m,), dtype=np.int64)
    q_t_tgt, q_t_cam, _ = sample_t_tgt_t_cam(
        rng=rng,
        queries_per_clip=m,
        clip_frames=t,
        prob_t_tgt_equals_t_cam=float(prob_t_tgt_equals_t_cam),
        q_t_src=q_t_src,
        t_src_tgt_delta_choices=t_src_tgt_delta_choices,
        t_src_tgt_delta_probs=t_src_tgt_delta_probs,
    )

    q_u = np.zeros((m,), dtype=np.float32)
    q_v = np.zeros((m,), dtype=np.float32)
    y_uv = np.zeros((m, 2), dtype=np.float32)
    y_xyz = np.zeros((m, 3), dtype=np.float32)
    y_disp = np.zeros((m, 3), dtype=np.float32)
    y_normal = np.zeros((m, 3), dtype=np.float32)
    y_vis = np.zeros((m,), dtype=np.float32)

    m_uv = np.zeros((m,), dtype=np.bool_)
    m_xyz = np.zeros((m,), dtype=np.bool_)
    m_disp = np.zeros((m,), dtype=np.bool_)
    m_vis = np.zeros((m,), dtype=np.bool_)
    m_normal = np.zeros((m,), dtype=np.bool_)
    is_hard_query = np.zeros((m,), dtype=np.bool_)

    t_cw_seq = np.full((t, 4, 4), np.nan, dtype=np.float32)
    for i in range(t):
        if not bool(camera_valid[i]):
            continue
        try:
            t_cw_seq[i] = np.linalg.inv(t_wc_seq[i]).astype(np.float32)
        except np.linalg.LinAlgError:
            continue

    k_inv_seq = np.full((t, 3, 3), np.nan, dtype=np.float32)
    for i in range(t):
        if not bool(camera_valid[i]):
            continue
        try:
            k_inv_seq[i] = np.linalg.inv(k_seq[i]).astype(np.float32)
        except np.linalg.LinAlgError:
            continue

    valid_pixels = [np.argwhere(depth_valid[idx] & bool(camera_valid[idx])) for idx in range(t)]
    hard_pixels = [np.argwhere(depth_boundary_mask(depth[idx], depth_valid[idx], q=0.9)) for idx in range(t)]

    # Pre-compute per-frame normal maps for pseudo surface normal supervision.
    normal_maps: list[np.ndarray] = []
    normal_valid_maps: list[np.ndarray] = []
    for idx in range(t):
        if bool(camera_valid[idx]) and np.isfinite(k_seq[idx]).all():
            n_map, n_valid = _compute_normal_map(depth[idx], k_seq[idx], depth_valid[idx])
        else:
            n_map = np.zeros((h, w, 3), dtype=np.float32)
            n_valid = np.zeros((h, w), dtype=bool)
        normal_maps.append(n_map)
        normal_valid_maps.append(n_valid)

    w_norm = max(1.0, float(w - 1))
    h_norm = max(1.0, float(h - 1))

    hard_target = int(sample_hard_query_flags(rng, m, float(hard_query_ratio)).sum())
    hard_eligible = np.array([i for i in range(m) if hard_pixels[int(q_t_src[i])].shape[0] > 0], dtype=np.int64)
    use_hard = np.zeros((m,), dtype=np.bool_)
    if hard_target > 0 and hard_eligible.size > 0:
        picked = rng.choice(hard_eligible, size=min(hard_target, hard_eligible.size), replace=False)
        use_hard[picked.astype(np.int64)] = True

    for i in range(m):
        fs = int(q_t_src[i])
        ft = int(q_t_tgt[i])
        fc = int(q_t_cam[i])
        if not (bool(camera_valid[fs]) and bool(camera_valid[ft]) and bool(camera_valid[fc])):
            continue

        src_candidates = valid_pixels[fs]
        picked_hard = bool(use_hard[i])
        if picked_hard:
            hp = hard_pixels[fs]
            if hp.shape[0] > 0:
                src_candidates = hp
            else:
                picked_hard = False
        if src_candidates.shape[0] == 0:
            continue

        pick = int(rng.integers(0, src_candidates.shape[0]))
        v_src = int(src_candidates[pick, 0])
        u_src = int(src_candidates[pick, 1])
        is_hard_query[i] = picked_hard

        z_src = float(depth[fs, v_src, u_src])
        if not np.isfinite(z_src) or z_src <= 0.0:
            continue

        pix_h = np.array([float(u_src), float(v_src), 1.0], dtype=np.float32)
        x_src_cam = k_inv_seq[fs] @ pix_h * z_src
        x_src_h = np.concatenate([x_src_cam, np.array([1.0], dtype=np.float32)], axis=0)
        x_world_h = t_wc_seq[fs] @ x_src_h

        x_cam_h = t_cw_seq[fc] @ x_world_h
        xyz_cam = x_cam_h[:3]
        if not np.isfinite(xyz_cam).all():
            continue

        q_u[i] = float(u_src) / w_norm
        q_v[i] = float(v_src) / h_norm
        y_xyz[i] = xyz_cam.astype(np.float32)
        m_vis[i] = True
        m_xyz[i] = True

        # Displacement: zero for static-scene depth reprojection.
        # When t_src == t_tgt, displacement is trivially zero (no time elapsed).
        # When t_src != t_tgt, defer m_disp to the depth-consistency check
        # below which confirms the point is still at the same world position.
        y_disp[i] = 0.0
        if fs == ft:
            m_disp[i] = True

        # Pseudo surface normal: look up from source frame, rotate to t_cam.
        if bool(normal_valid_maps[fs][v_src, u_src]):
            n_src = normal_maps[fs][v_src, u_src]  # in fs camera frame
            # Rotate: n_cam = R_cw_fc @ R_wc_fs @ n_src  (direction vector, no translation)
            r_wc_fs = t_wc_seq[fs, :3, :3]
            r_cw_fc = t_cw_seq[fc, :3, :3]
            n_cam = r_cw_fc @ (r_wc_fs @ n_src)
            n_len = float(np.linalg.norm(n_cam))
            if np.isfinite(n_cam).all() and n_len > 1e-6:
                y_normal[i] = (n_cam / n_len).astype(np.float32)
                m_normal[i] = True

        x_tgt_h = t_cw_seq[ft] @ x_world_h
        z_tgt = float(x_tgt_h[2])
        if not np.isfinite(z_tgt) or z_tgt <= 1e-6:
            continue
        proj = k_seq[ft] @ x_tgt_h[:3]
        u_tgt = float(proj[0] / z_tgt)
        v_tgt = float(proj[1] / z_tgt)
        if not (0.0 <= u_tgt <= (w - 1) and 0.0 <= v_tgt <= (h - 1)):
            continue

        u_nn = int(np.clip(round(u_tgt), 0, w - 1))
        v_nn = int(np.clip(round(v_tgt), 0, h - 1))
        if not bool(depth_valid[ft, v_nn, u_nn]):
            continue
        z_ref = float(depth[ft, v_nn, u_nn])
        if not np.isfinite(z_ref) or z_ref <= 0.0:
            continue
        if abs(z_ref - z_tgt) > max(0.05, 0.02 * z_ref):
            continue

        y_uv[i, 0] = np.clip(u_tgt / w_norm, 0.0, 1.0)
        y_uv[i, 1] = np.clip(v_tgt / h_norm, 0.0, 1.0)
        y_vis[i] = 1.0
        m_uv[i] = True
        # Depth consistency passed → point confirmed static at t_tgt.
        if fs != ft:
            m_disp[i] = True

    query = {"u": q_u, "v": q_v, "t_src": q_t_src, "t_tgt": q_t_tgt, "t_cam": q_t_cam}
    query_stats = {"is_hard_query": is_hard_query}
    target = {"xyz_3d": y_xyz, "uv_2d": y_uv, "visibility": y_vis, "displacement": y_disp, "normal": y_normal}
    mask = {"xyz_3d": m_xyz, "uv_2d": m_uv, "visibility": m_vis, "displacement": m_disp, "normal": m_normal}
    return query, target, mask, query_stats


def build_queries_from_trajectories(
    rng: np.random.Generator,
    traj_3d_world: np.ndarray,
    traj_visible: np.ndarray,
    traj_valid: np.ndarray,
    k_seq: np.ndarray,
    t_wc_seq: np.ndarray,
    camera_valid: np.ndarray,
    depth: np.ndarray | None,
    depth_valid: np.ndarray | None,
    queries_per_clip: int,
    hard_query_ratio: float,
    prob_t_tgt_equals_t_cam: float,
    t_src_tgt_delta_choices: tuple[int | None, ...] | None = None,
    t_src_tgt_delta_probs: tuple[float, ...] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build D4RT query/target/mask/query_stats from GT 3D trajectories.

    Unlike ``build_queries_from_depth`` which assumes static-scene geometry,
    this builder uses ground-truth 3D point trajectories so that displacement,
    visibility, and uv_2d are correct for dynamic objects.

    Args:
        traj_3d_world: [T, N, 3] world-coordinate 3D positions per point per frame.
        traj_visible:  [T, N] bool – GT visibility (not occluded & in-frame).
        traj_valid:    [T, N] bool – GT validity (tracking succeeded).
        k_seq:         [T, 3, 3] intrinsics (after crop/resize).
        t_wc_seq:      [T, 4, 4] camera-to-world transforms.
        camera_valid:  [T] bool.
        depth:         optional [T, H, W] for normal computation & hard-query sampling.
        depth_valid:   optional [T, H, W] for normal computation.
        t_src_tgt_delta_choices: optional local/global target timestep buckets relative to source time.
        t_src_tgt_delta_probs: optional probabilities for ``t_src_tgt_delta_choices``.
    """

    t_clip, n_pts, _ = traj_3d_world.shape
    m = int(queries_per_clip)

    # Need image size from depth or K to normalise pixel coords.
    if depth is not None:
        h, w = depth.shape[1], depth.shape[2]
    else:
        # Fallback: infer from K principal point (rough).
        h = int(round(float(k_seq[0, 1, 2]) * 2))
        w = int(round(float(k_seq[0, 0, 2]) * 2))
        h = max(h, 2)
        w = max(w, 2)

    # -- time sampling (reuse existing helpers) --------------------------------
    q_t_src = rng.integers(0, t_clip, size=(m,), dtype=np.int64)
    q_t_tgt, q_t_cam, _ = sample_t_tgt_t_cam(
        rng=rng,
        queries_per_clip=m,
        clip_frames=t_clip,
        prob_t_tgt_equals_t_cam=float(prob_t_tgt_equals_t_cam),
        q_t_src=q_t_src,
        t_src_tgt_delta_choices=t_src_tgt_delta_choices,
        t_src_tgt_delta_probs=t_src_tgt_delta_probs,
    )

    # -- output arrays ---------------------------------------------------------
    q_u = np.zeros((m,), dtype=np.float32)
    q_v = np.zeros((m,), dtype=np.float32)
    y_uv = np.zeros((m, 2), dtype=np.float32)
    y_xyz = np.zeros((m, 3), dtype=np.float32)
    y_disp = np.zeros((m, 3), dtype=np.float32)
    y_normal = np.zeros((m, 3), dtype=np.float32)
    y_vis = np.zeros((m,), dtype=np.float32)

    m_uv = np.zeros((m,), dtype=np.bool_)
    m_xyz = np.zeros((m,), dtype=np.bool_)
    m_disp = np.zeros((m,), dtype=np.bool_)
    m_vis = np.zeros((m,), dtype=np.bool_)
    m_normal = np.zeros((m,), dtype=np.bool_)
    is_hard_query = np.zeros((m,), dtype=np.bool_)

    # -- pre-compute T_cw for every frame --------------------------------------
    t_cw_seq = np.full((t_clip, 4, 4), np.nan, dtype=np.float32)
    for i in range(t_clip):
        if not bool(camera_valid[i]):
            continue
        try:
            t_cw_seq[i] = np.linalg.inv(t_wc_seq[i]).astype(np.float32)
        except np.linalg.LinAlgError:
            continue

    # -- per-frame eligible source candidates in crop-adjusted image space -----
    eligible_ids: list[np.ndarray] = []
    eligible_uv: list[np.ndarray] = []
    for fi in range(t_clip):
        if not bool(camera_valid[fi]):
            eligible_ids.append(np.zeros((0,), dtype=np.int64))
            eligible_uv.append(np.zeros((0, 2), dtype=np.float32))
            continue
        base_ok = (
            traj_valid[fi].astype(bool)
            & traj_visible[fi].astype(bool)
            & np.isfinite(traj_3d_world[fi]).all(axis=-1)
        )
        eids = np.where(base_ok)[0].astype(np.int64)
        if eids.size == 0:
            eligible_ids.append(eids)
            eligible_uv.append(np.zeros((0, 2), dtype=np.float32))
            continue

        pts_w = traj_3d_world[fi, eids]  # [E, 3]
        pts_h = np.concatenate([pts_w, np.ones((len(eids), 1), dtype=np.float32)], axis=1)
        pts_c = (t_cw_seq[fi] @ pts_h.T).T  # [E, 4]
        z_c = pts_c[:, 2]
        good = np.isfinite(z_c) & (z_c > 1e-6)
        proj = (k_seq[fi] @ pts_c[:, :3].T).T  # [E, 3]
        u_px = np.where(good, proj[:, 0] / np.maximum(z_c, 1e-8), -1.0)
        v_px = np.where(good, proj[:, 1] / np.maximum(z_c, 1e-8), -1.0)
        in_img = good & (u_px >= 0.0) & (u_px <= (w - 1)) & (v_px >= 0.0) & (v_px <= (h - 1))
        eligible_ids.append(eids[in_img])
        eligible_uv.append(np.stack([u_px[in_img], v_px[in_img]], axis=-1).astype(np.float32))

    # -- hard-query: project eligible points to image, find depth-boundary ones -
    hard_slots: list[np.ndarray] = []
    if depth is not None and depth_valid is not None:
        for fi in range(t_clip):
            if eligible_ids[fi].size == 0 or not bool(camera_valid[fi]):
                hard_slots.append(np.zeros((0,), dtype=np.int64))
                continue
            bmask = depth_boundary_mask(depth[fi], depth_valid[fi], q=0.9)
            if not bmask.any():
                hard_slots.append(np.zeros((0,), dtype=np.int64))
                continue
            src_uv = eligible_uv[fi]
            u_int = np.clip(np.rint(src_uv[:, 0]).astype(np.int64), 0, w - 1)
            v_int = np.clip(np.rint(src_uv[:, 1]).astype(np.int64), 0, h - 1)
            on_boundary = bmask[v_int, u_int]
            hard_slots.append(np.where(on_boundary)[0].astype(np.int64))
    else:
        hard_slots = [np.zeros((0,), dtype=np.int64) for _ in range(t_clip)]

    # -- decide which queries use hard sampling --------------------------------
    hard_target = int(sample_hard_query_flags(rng, m, float(hard_query_ratio)).sum())
    hard_eligible_q = np.array(
        [i for i in range(m) if hard_slots[int(q_t_src[i])].size > 0], dtype=np.int64,
    )
    use_hard = np.zeros((m,), dtype=np.bool_)
    if hard_target > 0 and hard_eligible_q.size > 0:
        picked = rng.choice(hard_eligible_q, size=min(hard_target, hard_eligible_q.size), replace=False)
        use_hard[picked.astype(np.int64)] = True

    # -- pre-compute normal maps (optional, from depth) ------------------------
    normal_maps: list[np.ndarray | None] = []
    normal_valid_maps: list[np.ndarray | None] = []
    if depth is not None and depth_valid is not None:
        for fi in range(t_clip):
            if bool(camera_valid[fi]) and np.isfinite(k_seq[fi]).all():
                n_map, n_valid = _compute_normal_map(depth[fi], k_seq[fi], depth_valid[fi])
            else:
                n_map = np.zeros((h, w, 3), dtype=np.float32)
                n_valid = np.zeros((h, w), dtype=bool)
            normal_maps.append(n_map)
            normal_valid_maps.append(n_valid)

    w_norm = max(1.0, float(w - 1))
    h_norm = max(1.0, float(h - 1))

    # -- main loop: build each query -------------------------------------------
    for i in range(m):
        fs = int(q_t_src[i])
        ft = int(q_t_tgt[i])
        fc = int(q_t_cam[i])
        if not (bool(camera_valid[fs]) and bool(camera_valid[ft]) and bool(camera_valid[fc])):
            continue

        # Pick a trajectory point from eligible set at source frame.
        if bool(use_hard[i]) and hard_slots[fs].size > 0:
            picked_slot = int(hard_slots[fs][int(rng.integers(0, hard_slots[fs].size))])
        else:
            if eligible_ids[fs].size == 0:
                continue
            picked_slot = int(rng.integers(0, eligible_ids[fs].size))
        if eligible_ids[fs].size == 0:
            continue
        pid = int(eligible_ids[fs][picked_slot])
        u_src = float(eligible_uv[fs][picked_slot, 0])
        v_src = float(eligible_uv[fs][picked_slot, 1])
        is_hard_query[i] = bool(use_hard[i]) and hard_slots[fs].size > 0

        # ---- source 2D (u, v) via projection of world point -----------------
        p_world_src = traj_3d_world[fs, pid]
        if not np.isfinite(p_world_src).all():
            continue
        p_src_h = np.array([*p_world_src, 1.0], dtype=np.float32)
        p_cam_src = t_cw_seq[fs] @ p_src_h
        z_src = float(p_cam_src[2])
        if not np.isfinite(z_src) or z_src <= 1e-6:
            continue
        if not (0.0 <= u_src <= (w - 1) and 0.0 <= v_src <= (h - 1)):
            continue

        q_u[i] = u_src / w_norm
        q_v[i] = v_src / h_norm

        # ---- target xyz_3d in t_cam coordinate frame -------------------------
        p_world_tgt = traj_3d_world[ft, pid]
        if not np.isfinite(p_world_tgt).all():
            continue
        p_tgt_h = np.array([*p_world_tgt, 1.0], dtype=np.float32)
        xyz_cam = (t_cw_seq[fc] @ p_tgt_h)[:3]
        if not np.isfinite(xyz_cam).all():
            continue

        y_xyz[i] = xyz_cam.astype(np.float32)
        m_xyz[i] = True

        # ---- displacement: P_world(t_tgt) - P_world(t_src) in t_cam frame ---
        delta_world = p_world_tgt - p_world_src
        r_cw_fc = t_cw_seq[fc, :3, :3]
        disp_cam = r_cw_fc @ delta_world
        if np.isfinite(disp_cam).all():
            y_disp[i] = disp_cam.astype(np.float32)
            m_disp[i] = bool(traj_valid[fs, pid]) and bool(traj_valid[ft, pid])

        # ---- target uv_2d + visibility in crop-adjusted target view ----------
        p_cam_tgt = (t_cw_seq[ft] @ p_tgt_h)[:3]
        z_tgt = float(p_cam_tgt[2])
        target_in_frame = False
        u_tgt = 0.0
        v_tgt = 0.0
        if np.isfinite(z_tgt) and z_tgt > 1e-6:
            proj_tgt = k_seq[ft] @ p_cam_tgt
            u_tgt = float(proj_tgt[0] / z_tgt)
            v_tgt = float(proj_tgt[1] / z_tgt)
            target_in_frame = (
                np.isfinite(u_tgt)
                and np.isfinite(v_tgt)
                and 0.0 <= u_tgt <= (w - 1)
                and 0.0 <= v_tgt <= (h - 1)
            )

        # GT visibility encodes dataset occlusion/full-frame visibility. After
        # random crop, a full-frame visible target must also project into the
        # crop-adjusted target image to be visible to the model.
        target_visible = (
            bool(traj_valid[ft, pid])
            and bool(traj_visible[ft, pid])
            and target_in_frame
        )
        if bool(traj_valid[ft, pid]):
            y_vis[i] = 1.0 if target_visible else 0.0
            m_vis[i] = True

        # Only supervise 2D target location when the target is genuinely visible
        # in the crop-adjusted target view.
        if target_visible:
            y_uv[i, 0] = np.clip(u_tgt / w_norm, 0.0, 1.0)
            y_uv[i, 1] = np.clip(v_tgt / h_norm, 0.0, 1.0)
            m_uv[i] = True

        # ---- normal from depth (optional) ------------------------------------
        if normal_maps and normal_valid_maps:
            u_src_int = int(np.clip(round(u_src), 0, w - 1))
            v_src_int = int(np.clip(round(v_src), 0, h - 1))
            n_valid_map = normal_valid_maps[fs]
            if n_valid_map is not None and bool(n_valid_map[v_src_int, u_src_int]):
                n_src = normal_maps[fs][v_src_int, u_src_int]
                r_wc_fs = t_wc_seq[fs, :3, :3]
                n_cam = r_cw_fc @ (r_wc_fs @ n_src)
                n_len = float(np.linalg.norm(n_cam))
                if np.isfinite(n_cam).all() and n_len > 1e-6:
                    y_normal[i] = (n_cam / n_len).astype(np.float32)
                    m_normal[i] = True

    query = {"u": q_u, "v": q_v, "t_src": q_t_src, "t_tgt": q_t_tgt, "t_cam": q_t_cam}
    query_stats = {"is_hard_query": is_hard_query}
    target = {"xyz_3d": y_xyz, "uv_2d": y_uv, "visibility": y_vis, "displacement": y_disp, "normal": y_normal}
    mask = {"xyz_3d": m_xyz, "uv_2d": m_uv, "visibility": m_vis, "displacement": m_disp, "normal": m_normal}
    return query, target, mask, query_stats
