import torch

from lam.integrations.gld import extract_gld_f0_f1_tokens


def test_extract_gld_f0_f1_tokens_strips_cls_token():
    features = {
        0: torch.randn(2, 257, 1536),
        1: torch.randn(2, 257, 1536),
    }

    f0, f1 = extract_gld_f0_f1_tokens(features, batch_size=2, expected_tokens=256)

    assert f0.shape == (2, 256, 1536)
    assert f1.shape == (2, 256, 1536)
    assert torch.equal(f0, features[0][:, 1:])
    assert torch.equal(f1, features[1][:, 1:])


def test_extract_gld_f0_f1_tokens_rejects_wrong_shape():
    features = {
        0: torch.randn(2, 128, 1536),
        1: torch.randn(2, 256, 1536),
    }

    try:
        extract_gld_f0_f1_tokens(features, batch_size=2, expected_tokens=256)
    except ValueError as exc:
        assert "expected 256 patch tokens" in str(exc)
    else:
        raise AssertionError("expected ValueError")
