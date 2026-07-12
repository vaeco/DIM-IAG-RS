import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from .dim import DIM


def pad_to_multiple(x, multiple=8):
    h, w = x.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    if pad_h == 0 and pad_w == 0:
        return x, (h, w)

    # reflect padding requires input spatial size > padding.
    # For normal images this is fine.
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, (h, w)


class HaarDWT(nn.Module):
    """
    Pure PyTorch Haar DWT.
    Input:  B, C, H, W
    Output: B, 4C, H/2, W/2
    """
    def forward(self, x):
        h, w = x.shape[-2:]
        pad_h = h % 2
        pad_w = w % 2

        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 - x01 + x10 - x11) * 0.5
        hl = (x00 + x01 - x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5

        return torch.cat([ll, lh, hl, hh], dim=1), (h, w)


class HaarIWT(nn.Module):
    """
    Inverse Haar transform.
    Input:  B, 4C, H/2, W/2
    Output: B, C, H, W
    """
    def forward(self, coeff, out_size):
        h_out, w_out = out_size
        b, c4, h, w = coeff.shape
        c = c4 // 4

        ll, lh, hl, hh = torch.chunk(coeff, 4, dim=1)

        x00 = (ll + lh + hl + hh) * 0.5
        x01 = (ll - lh + hl - hh) * 0.5
        x10 = (ll + lh - hl - hh) * 0.5
        x11 = (ll - lh - hl + hh) * 0.5

        x = torch.zeros(b, c, h * 2, w * 2, device=coeff.device, dtype=coeff.dtype)
        x[:, :, 0::2, 0::2] = x00
        x[:, :, 0::2, 1::2] = x01
        x[:, :, 1::2, 0::2] = x10
        x[:, :, 1::2, 1::2] = x11

        return x[:, :, :h_out, :w_out]


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.body(x)


class WaveletResidualRefiner(nn.Module):
    """
    Lightweight wavelet-domain residual refinement.

    It receives wavelet coefficients of:
    1) original low-light image
    2) DIM enhanced image

    Then it predicts wavelet residual coefficients.
    The final RGB residual is reconstructed by inverse Haar transform.
    """
    def __init__(self, in_rgb_channels=3, hidden=32, num_blocks=3, residual_scale=0.1):
        super().__init__()
        self.dwt = HaarDWT()
        self.iwt = HaarIWT()

        # low-light coeff: 4*3, DIM coeff: 4*3, total 24 channels
        in_ch = in_rgb_channels * 4 * 2
        out_ch = in_rgb_channels * 4

        layers = [
            nn.Conv2d(in_ch, hidden, 3, 1, 1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_blocks):
            layers.append(ResBlock(hidden))
        layers.append(nn.Conv2d(hidden, out_ch, 3, 1, 1))

        self.net = nn.Sequential(*layers)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

        # Zero init: initial WRR residual is 0.
        # Therefore DIM_WRR initially behaves like original pretrained DIM.
        last_conv = self.net[-1]
        nn.init.zeros_(last_conv.weight)
        nn.init.zeros_(last_conv.bias)

    def forward(self, low_img, dim_img):
        low_img = torch.clamp(low_img, 0.0, 1.0)
        dim_img = torch.clamp(dim_img, 0.0, 1.0)

        low_coeff, size = self.dwt(low_img)
        dim_coeff, _ = self.dwt(dim_img)

        feat = torch.cat([low_coeff, dim_coeff], dim=1)
        delta_coeff = self.net(feat)
        residual = self.iwt(delta_coeff, size)

        return residual * self.residual_scale


class DIM_WRR(nn.Module):
    """
    DIM + Wavelet Residual Refinement.

    The original DIM formula is not modified.
    The wavelet branch only learns a small residual after DIM.

    This wrapper pads arbitrary input sizes to multiples of 8 before DIM,
    then crops the output back to the original size.
    """
    def __init__(self, c1=3, c_hidden=32, refiner_hidden=32, refiner_blocks=3):
        super().__init__()
        self.dim = DIM(c1=c1, c_hidden=c_hidden)
        self.refiner = WaveletResidualRefiner(
            in_rgb_channels=3,
            hidden=refiner_hidden,
            num_blocks=refiner_blocks,
            residual_scale=0.1,
        )

    def forward(self, x, return_base=False):
        x_pad, (h, w) = pad_to_multiple(x, multiple=8)

        dim_out_pad = self.dim(x_pad)
        residual_pad = self.refiner(x_pad, dim_out_pad)
        out_pad = torch.clamp(dim_out_pad + residual_pad, 0.0, 1.0)

        out = out_pad[:, :, :h, :w]
        dim_out = dim_out_pad[:, :, :h, :w]

        if return_base:
            return out, dim_out
        return out


def load_dim_checkpoint_into_wrr(model, ckpt_path, map_location="cpu"):
    """
    Load original DIM checkpoint into model.dim.
    Works for direct state_dict or common checkpoint dict formats.
    """
    ckpt = torch.load(ckpt_path, map_location=map_location)

    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "params"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    state = {}
    for k, v in ckpt.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]
        if nk.startswith("dim."):
            nk = nk[len("dim."):]
        state[nk] = v

    missing, unexpected = model.dim.load_state_dict(state, strict=False)

    print("[load_dim_checkpoint_into_wrr]")
    print("  ckpt:", ckpt_path)
    print("  missing keys:", len(missing))
    print("  unexpected keys:", len(unexpected))
    if len(missing) < 20:
        print("  missing:", missing)
    if len(unexpected) < 20:
        print("  unexpected:", unexpected)

    return model


if __name__ == "__main__":
    # Test both regular and irregular image sizes.
    model = DIM_WRR()

    for shape in [(1, 3, 512, 512), (1, 3, 513, 769), (1, 3, 640, 960)]:
        x = torch.rand(*shape)
        y, y0 = model(x, return_base=True)
        print("input:", x.shape, "output:", y.shape, "base:", y0.shape)
