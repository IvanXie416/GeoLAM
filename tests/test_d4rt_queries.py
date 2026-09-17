import torch

from lam.integrations.d4rt import build_d4rt_clip_queries, build_d4rt_pair_queries


def test_build_d4rt_pair_queries_uses_fixed_32x32_grid():
    q_cur, q_fut = build_d4rt_pair_queries(batch_size=2, grid_size=32, device=torch.device("cpu"))

    assert q_cur["u"].shape == (2, 1024)
    assert q_cur["v"].shape == (2, 1024)
    assert torch.all(q_cur["u"] >= 0)
    assert torch.all(q_cur["u"] <= 1)
    assert torch.all(q_cur["v"] >= 0)
    assert torch.all(q_cur["v"] <= 1)
    assert torch.equal(q_cur["u"], q_fut["u"])
    assert torch.equal(q_cur["v"], q_fut["v"])


def test_build_d4rt_pair_queries_sets_current_and_future_time_fields():
    q_cur, q_fut = build_d4rt_pair_queries(batch_size=1, grid_size=32, device=torch.device("cpu"))

    assert torch.equal(q_cur["t_src"], torch.zeros(1, 1024, dtype=torch.long))
    assert torch.equal(q_cur["t_tgt"], torch.zeros(1, 1024, dtype=torch.long))
    assert torch.equal(q_cur["t_cam"], torch.zeros(1, 1024, dtype=torch.long))
    assert torch.equal(q_fut["t_src"], torch.zeros(1, 1024, dtype=torch.long))
    assert torch.equal(q_fut["t_tgt"], torch.ones(1, 1024, dtype=torch.long))
    assert torch.equal(q_fut["t_cam"], torch.zeros(1, 1024, dtype=torch.long))


def test_build_d4rt_clip_queries_uses_explicit_context_times():
    q_cur, q_fut = build_d4rt_clip_queries(
        batch_size=2,
        grid_size=2,
        device=torch.device("cpu"),
        t_src=torch.tensor([0, 1]),
        t_tgt=torch.tensor([5, 4]),
        t_cam=torch.tensor([0, 1]),
    )

    assert torch.equal(q_cur["t_src"][:, 0], torch.tensor([0, 1]))
    assert torch.equal(q_cur["t_tgt"][:, 0], torch.tensor([0, 1]))
    assert torch.equal(q_fut["t_src"][:, 0], torch.tensor([0, 1]))
    assert torch.equal(q_fut["t_tgt"][:, 0], torch.tensor([5, 4]))
    assert torch.equal(q_fut["t_cam"][:, 0], torch.tensor([0, 1]))
