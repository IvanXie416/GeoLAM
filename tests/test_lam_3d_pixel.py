import pytest
import torch

from lam.models.lam_3d_pixel import LAM3DPixel, RGBPatchDecoder


def test_rgb_patch_decoder_reconstructs_channel_first_image_and_backpropagates():
    decoder = RGBPatchDecoder(
        token_dim=32,
        image_size=32,
        patch_size=2,
        heads=4,
        depth=1,
    )
    tokens = torch.randn(2, 256, 32, requires_grad=True)

    rgb_hat = decoder(tokens)

    assert rgb_hat.shape == (2, 3, 32, 32)
    assert torch.all((0.0 <= rgb_hat) & (rgb_hat <= 1.0))
    rgb_hat.mean().backward()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()


def test_rgb_patch_decoder_rejects_wrong_token_grid():
    decoder = RGBPatchDecoder(token_dim=32, image_size=32, patch_size=2, heads=4, depth=1)

    with pytest.raises(ValueError, match="expects 256 tokens"):
        decoder(torch.randn(1, 255, 32))


def test_lam_3d_pixel_keeps_gld_outputs_and_adds_rgb_prediction():
    model = LAM3DPixel(
        in_dim=24,
        c_state=32,
        c_z=32,
        z_dim=16,
        c_geo=5,
        k_action=1,
        k_geo=1,
        latent_dim=4,
        heads=4,
        action_bottleneck_depth=1,
        idm_depth=1,
        fdm_depth=1,
        image_size=32,
        rgb_patch_size=2,
        rgb_decoder_depth=1,
    )
    features = [torch.randn(1, 256, 24) for _ in range(4)]

    output = model(*features)

    assert output.pred_tokens_tk.shape == (1, 256, 32)
    assert output.f0_hat.shape == (1, 256, 24)
    assert output.f1_hat.shape == (1, 256, 24)
    assert output.rgb_hat.shape == (1, 3, 32, 32)
    assert output.latent_action.low_dim.shape == (1, 1, 4)
