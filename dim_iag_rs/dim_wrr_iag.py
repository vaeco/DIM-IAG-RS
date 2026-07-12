import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from .dim_wrr import DIM_WRR


def logit(p):
    p = min(max(float(p), 1e-4), 1.0 - 1e-4)
    return math.log(p / (1.0 - p))


def luminance(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def sobel_edge_from_luma(luma):
    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=luma.dtype,
        device=luma.device,
    ).view(1, 1, 3, 3)

    ky = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=luma.dtype,
        device=luma.device,
    ).view(1, 1, 3, 3)

    gx = F.conv2d(luma, kx, padding=1)
    gy = F.conv2d(luma, ky, padding=1)
    edge = torch.sqrt(gx * gx + gy * gy + 1e-6)
    return torch.clamp(edge, 0.0, 1.0)


def high_frequency_proxy(luma, kernel_size=5):
    pad = kernel_size // 2
    smooth = F.avg_pool2d(luma, kernel_size=kernel_size, stride=1, padding=pad)
    return torch.clamp(torch.abs(luma - smooth), 0.0, 1.0)


class IlluminationEstimator(nn.Module):
    """
    Estimate illumination map I from:
        X(3) + luma(1) + edge(1) = 5 channels
    Output:
        I in [0,1], shape B,1,H,W
    """
    def __init__(self, hidden=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(5, hidden, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x, luma=None, edge=None):
        x = torch.clamp(x, 0.0, 1.0)
        if luma is None:
            luma = luminance(x)
        if edge is None:
            edge = sobel_edge_from_luma(luma)
        feat = torch.cat([x, luma, edge], dim=1)
        return self.net(feat)


class NoiseEstimator(nn.Module):
    """
    Estimate noise/degradation map N from:
        X(3) + luma(1) + edge(1) + residual_mag(1) + high_freq(1) = 7 channels
    Output:
        N in [0,1], shape B,1,H,W
    """
    def __init__(self, hidden=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(7, hidden, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x, luma=None, edge=None, residual_mag=None):
        x = torch.clamp(x, 0.0, 1.0)
        if luma is None:
            luma = luminance(x)
        if edge is None:
            edge = sobel_edge_from_luma(luma)
        if residual_mag is None:
            residual_mag = torch.zeros_like(luma)

        hf = high_frequency_proxy(luma)
        feat = torch.cat([x, luma, edge, residual_mag, hf], dim=1)
        return self.net(feat)


class IlluminationNoiseAwareGate(nn.Module):
    """
    Gate input:
        X(3)
        Y_base(3)
        R_proxy(3)
        I(1)
        N(1)
        luma(1)
        edge(1)
        residual_mag(1)
        under_exposure_map(1)
        over_exposure_risk(1)

    Total: 16 channels

    Output:
        G in [0,1], shape B,1,H,W
    """
    def __init__(self, hidden=24, init_gate=0.45):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(16, hidden, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1, 1, 0),
            nn.Sigmoid(),
        )

        # Important: start from a conservative gate.
        last_conv = self.net[-2]
        nn.init.zeros_(last_conv.weight)
        nn.init.constant_(last_conv.bias, logit(init_gate))

    def forward(self, x, y_base, r_proxy, illum, noise, luma, edge, residual_mag):
        x = torch.clamp(x, 0.0, 1.0)
        y_base = torch.clamp(y_base, 0.0, 1.0)
        r_proxy = torch.clamp(r_proxy, 0.0, 1.0)
        illum = torch.clamp(illum, 0.0, 1.0)
        noise = torch.clamp(noise, 0.0, 1.0)

        under_exposure = F.relu(0.5 - luma)
        over_exposure_risk = F.relu(y_base - 0.95).mean(dim=1, keepdim=True)

        feat = torch.cat(
            [
                x,
                y_base,
                r_proxy,
                illum,
                noise,
                luma,
                edge,
                residual_mag,
                under_exposure,
                over_exposure_risk,
            ],
            dim=1,
        )
        return self.net(feat)


class DIM_WRR_IAG(nn.Module):
    """
    DIM-WRR + Illumination/Noise-Aware Gate.

    Base:
        y_wrr, y_dim = DIM_WRR(x, return_base=True)
        y_base = clamp(y_dim + base_alpha * (y_wrr - y_dim), 0, 1)

    IAG:
        I = IlluminationEstimator(x)
        N = NoiseEstimator(x, residual_mag)
        R = clamp(x / (I + eps), 0, 1)
        G = Gate(x, y_base, R, I, N, luma, edge, residual_mag)
        y = clamp(x + G * (y_base - x), 0, 1)
    """
    def __init__(
        self,
        c1=3,
        c_hidden=32,
        refiner_hidden=32,
        refiner_blocks=3,
        gate_hidden=24,
        illum_hidden=24,
        noise_hidden=24,
        base_alpha=0.6,
        init_gate=0.45,
        eps=1e-3,
    ):
        super().__init__()

        self.base = DIM_WRR(
            c1=c1,
            c_hidden=c_hidden,
            refiner_hidden=refiner_hidden,
            refiner_blocks=refiner_blocks,
        )

        self.illumination = IlluminationEstimator(hidden=illum_hidden)
        self.noise = NoiseEstimator(hidden=noise_hidden)
        self.gate = IlluminationNoiseAwareGate(
            hidden=gate_hidden,
            init_gate=init_gate,
        )

        self.base_alpha = float(base_alpha)
        self.eps = float(eps)

    def freeze_base(self):
        for p in self.base.parameters():
            p.requires_grad = False

    def unfreeze_base(self):
        for p in self.base.parameters():
            p.requires_grad = True

    def unfreeze_refiner_only(self):
        for p in self.base.parameters():
            p.requires_grad = False

        # Compatible with common attribute names.
        for name, module in self.base.named_modules():
            if "refiner" in name.lower():
                for p in module.parameters():
                    p.requires_grad = True

    def make_base_enhancement(self, x):
        y_wrr, y_dim = self.base(x, return_base=True)
        residual = y_wrr - y_dim
        y_base = torch.clamp(y_dim + self.base_alpha * residual, 0.0, 1.0)
        return y_base, y_wrr, y_dim

    def forward(self, x, return_gate=False, return_all=False):
        x = torch.clamp(x, 0.0, 1.0)

        y_base, y_wrr, y_dim = self.make_base_enhancement(x)

        luma = luminance(x)
        edge = sobel_edge_from_luma(luma)
        residual_mag = torch.mean(torch.abs(y_base - x), dim=1, keepdim=True)

        illum = self.illumination(x, luma=luma, edge=edge)
        noise = self.noise(x, luma=luma, edge=edge, residual_mag=residual_mag)

        r_proxy = torch.clamp(x / (illum + self.eps), 0.0, 1.0)

        gate = self.gate(
            x=x,
            y_base=y_base,
            r_proxy=r_proxy,
            illum=illum,
            noise=noise,
            luma=luma,
            edge=edge,
            residual_mag=residual_mag,
        )

        y = torch.clamp(x + gate * (y_base - x), 0.0, 1.0)

        if return_all:
            aux = {
                "gate": gate,
                "illumination": illum,
                "noise": noise,
                "reflectance": r_proxy,
                "y_base": y_base,
                "y_wrr": y_wrr,
                "y_dim": y_dim,
                "luma": luma,
                "edge": edge,
                "residual_mag": residual_mag,
            }
            return y, aux

        if return_gate:
            return y, gate

        return y


def _strip_module_prefix(state):
    new_state = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        new_state[nk] = v
    return new_state


def load_wrr_checkpoint_into_iag(model, ckpt_path, map_location="cpu"):
    """
    Load old DIM-WRR checkpoint into model.base.
    This is used before training IAG gate / illumination / noise modules.
    """
    ckpt = torch.load(ckpt_path, map_location=map_location)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    state = _strip_module_prefix(state)
    missing, unexpected = model.base.load_state_dict(state, strict=False)

    print("[load_wrr_checkpoint_into_iag]")
    print("ckpt:", ckpt_path)
    print("missing:", len(missing))
    print("unexpected:", len(unexpected))
    if len(missing) > 0:
        print("first missing:", missing[:10])
    if len(unexpected) > 0:
        print("first unexpected:", unexpected[:10])

    return model


def load_iag_checkpoint(model, ckpt_path, map_location="cpu", strict=False):
    """
    Load full IAG checkpoint.
    Used for enhancement/evaluation after training.
    """
    ckpt = torch.load(ckpt_path, map_location=map_location)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    state = _strip_module_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=strict)

    print("[load_iag_checkpoint]")
    print("ckpt:", ckpt_path)
    print("missing:", len(missing))
    print("unexpected:", len(unexpected))
    if len(missing) > 0:
        print("first missing:", missing[:10])
    if len(unexpected) > 0:
        print("first unexpected:", unexpected[:10])

    return model


if __name__ == "__main__":
    model = DIM_WRR_IAG(base_alpha=0.6, init_gate=0.45)
    x = torch.rand(1, 3, 513, 769)

    with torch.no_grad():
        y, aux = model(x, return_all=True)

    print("x:", x.shape)
    print("y:", y.shape)
    print("gate:", aux["gate"].shape, float(aux["gate"].min()), float(aux["gate"].max()), float(aux["gate"].mean()))
    print("illumination:", aux["illumination"].shape, float(aux["illumination"].min()), float(aux["illumination"].max()), float(aux["illumination"].mean()))
    print("noise:", aux["noise"].shape, float(aux["noise"].min()), float(aux["noise"].max()), float(aux["noise"].mean()))
    print("reflectance:", aux["reflectance"].shape)
    print("y_base:", aux["y_base"].shape)
    print("y_wrr:", aux["y_wrr"].shape)
    print("y_dim:", aux["y_dim"].shape)
