from pathlib import Path

import torch

import lam.predict_rgb as predict_rgb
from lam.predict_rgb import (
    decode_predicted_rgb,
    decode_training_consistent_rgb,
    parse_args,
    tokens_to_feature_map,
)


def test_parse_args_for_action_transfer_rgb_prediction():
    args = parse_args(
        [
            "--checkpoint",
            "outputs/lam_last.pt",
            "--action-src",
            "action_t.png",
            "--action-tgt",
            "action_tk.png",
            "--condition",
            "prime_t.png",
            "--output-dir",
            "predictions",
        ]
    )

    assert args.checkpoint == Path("outputs/lam_last.pt")
    assert args.action_src == Path("action_t.png")
    assert args.action_tgt == Path("action_tk.png")
    assert args.condition == Path("prime_t.png")
    assert args.output_dir == Path("predictions")
    assert args.decoder_config == Path("configs/gld_mae_decoder")


def test_tokens_to_feature_map_reshapes_square_patch_tokens():
    tokens = torch.randn(2, 16, 8)

    feature_map = tokens_to_feature_map(tokens)

    assert feature_map.shape == (2, 8, 4, 4)


def test_tokens_to_feature_map_rejects_non_square_token_count():
    tokens = torch.randn(2, 18, 8)

    try:
        tokens_to_feature_map(tokens)
    except ValueError as exc:
        assert "square" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_training_consistent_decode_does_not_renormalize_predicted_features(monkeypatch):
    f0_hat = torch.randn(1, 4, 8)
    f1_hat = torch.randn(1, 4, 8)
    expected = torch.randn(1, 3, 28, 28)
    captured = {}

    def fake_decode(gld, f0, f1, height, width):
        captured.update(gld=gld, f0=f0, f1=f1, height=height, width=width)
        return expected

    monkeypatch.setattr(predict_rgb, "decode_predicted_rgb", fake_decode)
    gld = object()

    result = decode_training_consistent_rgb(gld, f0_hat, f1_hat, image_size=28)

    assert result is expected
    assert captured["gld"] is gld
    assert captured["f0"] is f0_hat
    assert captured["f1"] is f1_hat
    assert captured["height"] == 28
    assert captured["width"] == 28


class _FakeGLDModel:
    def __init__(self, *, do_normalization=False):
        self.mae_decoder = object()
        self.do_normalization = do_normalization
        self.encoder_mean = torch.zeros(1, 3, 1, 1)
        self.encoder_std = torch.ones(1, 3, 1, 1)
        self.propagation_input = None
        self.decode_features = None

    def _normalize(self, patches, cls):
        return patches + 100.0, cls + 200.0

    def propagate_features(self, features, from_level, total_view, cls_token, normalize_patches):
        self.propagation_input = (features, from_level, total_view, cls_token, normalize_patches)
        batch, channels, height, width = features.shape
        patches = features.reshape(batch, channels, height * width).transpose(1, 2).unsqueeze(1)
        cls = cls_token.unsqueeze(1)
        return [(patches + offset, cls) for offset in (1.0, 2.0, 3.0)]

    def decode(self, features, H, W):
        self.decode_features = features
        return {"rgb": torch.full((features[0][0].shape[0], 3, H, W), 0.5)}


class _FakeGLD:
    def __init__(self, *, do_normalization=False):
        self.model = _FakeGLDModel(do_normalization=do_normalization)


def test_decode_uses_raw_l0_and_propagated_raw_l1_l3():
    gld = _FakeGLD()
    f0 = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)
    f1 = f0 + 50.0

    rgb = decode_predicted_rgb(gld, f0, f1, 28, 28)

    propagation_features, from_level, total_view, propagation_cls, normalize_patches = gld.model.propagation_input
    assert torch.equal(propagation_features, tokens_to_feature_map(f1))
    assert torch.count_nonzero(propagation_cls) == 0
    assert from_level == 1
    assert total_view == 1
    assert normalize_patches is False
    decoded = gld.model.decode_features
    assert torch.equal(decoded[0][0][:, 0], f0)
    assert torch.equal(decoded[1][0], propagation_features.flatten(2).transpose(1, 2).unsqueeze(1) + 1.0)
    assert rgb.shape == (1, 3, 28, 28)


def test_decode_normalizes_raw_l1_when_gld_stats_are_enabled():
    gld = _FakeGLD(do_normalization=True)
    f0 = torch.randn(1, 4, 8)
    f1 = torch.randn(1, 4, 8)

    decode_predicted_rgb(gld, f0, f1, 28, 28)

    propagation_features, _, _, propagation_cls, normalize_patches = gld.model.propagation_input
    assert torch.equal(propagation_features, tokens_to_feature_map(f1) + 100.0)
    assert torch.equal(propagation_cls, torch.full_like(propagation_cls, 200.0))
    assert normalize_patches is False
