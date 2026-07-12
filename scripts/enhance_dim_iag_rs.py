"""Final DIM-IAG-RS inference with a dual-gate illumination router.

The DIM-WRR backbone is shared by two complementary IAG experts:
  1. a conservative expert for moderately illuminated inputs;
  2. a brightness-oriented expert for severely under-exposed inputs.

The image-level router computes a continuous mixing coefficient rho from
under-exposure severity and exposure/noise risk. The final effective gate is

    E = (1 - rho) * E_conservative + rho * E_bright

and the enhanced image is

    Y = clamp(X + E * (Y_base - X), 0, 1).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dim_iag_rs.dim_wrr_iag import DIM_WRR_IAG, load_iag_checkpoint


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


def image_luma(x: torch.Tensor) -> torch.Tensor:
    """Compute ITU-R BT.601 luminance for a BCHW RGB tensor."""
    return (
        0.299 * x[:, 0:1]
        + 0.587 * x[:, 1:2]
        + 0.114 * x[:, 2:3]
    )


def make_under_map(x: torch.Tensor, threshold: float) -> torch.Tensor:
    """Normalized under-exposure severity map in [0, 1]."""
    luma = image_luma(x)
    under = F.relu(float(threshold) - luma) / max(float(threshold), 1e-6)
    return under.clamp(0.0, 1.0)


def make_over_map(x: torch.Tensor, threshold: float) -> torch.Tensor:
    """Normalized over-exposure risk map in [0, 1]."""
    over = F.relu(x - float(threshold)).mean(dim=1, keepdim=True)
    over = over / max(1.0 - float(threshold), 1e-6)
    return over.clamp(0.0, 1.0)


def list_images(
    input_dir: Path,
    max_images: int = 0,
    balanced_subdirs: bool = False,
) -> List[Path]:
    """List images recursively while preserving a stable deterministic order."""
    all_images = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if max_images <= 0:
        return all_images

    if not balanced_subdirs:
        return all_images[:max_images]

    subdirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    if not subdirs:
        return all_images[:max_images]

    base_count = max_images // len(subdirs)
    remainder = max_images % len(subdirs)
    selected: List[Path] = []

    for index, subdir in enumerate(subdirs):
        current_count = base_count + int(index < remainder)
        current_images = sorted(
            path
            for path in subdir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        selected.extend(current_images[:current_count])

    return selected[:max_images]


def resize_if_needed(x: torch.Tensor, max_long: int) -> torch.Tensor:
    """Optionally resize so that the longer side does not exceed max_long."""
    if max_long <= 0:
        return x

    height, width = x.shape[-2:]
    long_side = max(height, width)
    if long_side <= max_long:
        return x

    scale = float(max_long) / float(long_side)
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))
    return F.interpolate(
        x,
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
    )


def save_tensor_image(image_tensor: torch.Tensor, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = TF.to_pil_image(image_tensor.detach().cpu().clamp(0.0, 1.0))
    image.save(output_path)


def average(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA is unavailable; falling back to CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(device_arg)


def validate_path(path: Path, kind: str) -> None:
    if kind == "file" and not path.is_file():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    if kind == "dir" and not path.is_dir():
        raise FileNotFoundError(f"Required directory does not exist: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DIM-IAG-RS final dual-gate router inference"
    )

    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--conservative_ckpt",
        default="weights/dim_iag_rs_best.pth",
        help="Full conservative IAG checkpoint.",
    )
    parser.add_argument(
        "--bright_ckpt",
        default="weights/dim_iag_rs_bright_gate_best.pth",
        help="Brightness-oriented IAG checkpoint.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--base_alpha", type=float, default=0.6)
    parser.add_argument("--init_gate", type=float, default=0.45)

    # Final router parameters validated in the current release.
    parser.add_argument("--router_center", type=float, default=0.81)
    parser.add_argument("--router_width", type=float, default=0.02)
    parser.add_argument("--rho_max", type=float, default=0.95)
    parser.add_argument("--router_under_th", type=float, default=0.35)
    parser.add_argument("--router_over_th", type=float, default=0.88)
    parser.add_argument("--risk_over_weight", type=float, default=1.50)
    parser.add_argument("--risk_noise_weight", type=float, default=0.50)
    parser.add_argument("--risk_floor", type=float, default=0.50)
    parser.add_argument("--effective_max", type=float, default=1.50)

    parser.add_argument("--max_long", type=int, default=0)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--balanced_subdirs", action="store_true")
    parser.add_argument("--print_freq", type=int, default=100)
    return parser.parse_args()


def build_model(
    checkpoint: Path,
    device: torch.device,
    base_alpha: float,
    init_gate: float,
    expert_name: str,
) -> DIM_WRR_IAG:
    print(f"========== Load {expert_name} expert ==========", flush=True)
    model = DIM_WRR_IAG(
        base_alpha=base_alpha,
        init_gate=init_gate,
    ).to(device).eval()
    load_iag_checkpoint(
        model,
        str(checkpoint),
        map_location=device,
        strict=False,
    )
    return model


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    conservative_ckpt = Path(args.conservative_ckpt).expanduser().resolve()
    bright_ckpt = Path(args.bright_ckpt).expanduser().resolve()

    validate_path(input_dir, "dir")
    validate_path(conservative_ckpt, "file")
    validate_path(bright_ckpt, "file")

    device = resolve_device(args.device)
    print("device:", device, flush=True)

    conservative_model = build_model(
        checkpoint=conservative_ckpt,
        device=device,
        base_alpha=args.base_alpha,
        init_gate=args.init_gate,
        expert_name="conservative",
    )
    bright_model = build_model(
        checkpoint=bright_ckpt,
        device=device,
        base_alpha=args.base_alpha,
        init_gate=args.init_gate,
        expert_name="bright",
    )

    image_files = list_images(
        input_dir=input_dir,
        max_images=args.max_images,
        balanced_subdirs=args.balanced_subdirs,
    )
    if not image_files:
        raise RuntimeError(f"No images found in: {input_dir}")

    print("========== Dual-Gate Router Config ==========", flush=True)
    print("method: DIM-IAG-RS Dual-Gate Router", flush=True)
    print("input_dir:", input_dir, flush=True)
    print("output_dir:", output_dir, flush=True)
    print("conservative_ckpt:", conservative_ckpt, flush=True)
    print("bright_ckpt:", bright_ckpt, flush=True)
    print("num_images:", len(image_files), flush=True)
    print("base_alpha:", args.base_alpha, flush=True)
    print("router_center:", args.router_center, flush=True)
    print("router_width:", args.router_width, flush=True)
    print("rho_max:", args.rho_max, flush=True)
    print("router_under_th:", args.router_under_th, flush=True)
    print("router_over_th:", args.router_over_th, flush=True)

    stats: Dict[str, List[float]] = {
        "gate_c": [],
        "gate_b": [],
        "e_c": [],
        "e_b": [],
        "router_under": [],
        "router_over": [],
        "rho": [],
        "effective": [],
        "ybase_diff": [],
        "input_luma": [],
        "output_luma": [],
    }

    with torch.inference_mode():
        for index, image_path in enumerate(image_files, start=1):
            image = Image.open(image_path).convert("RGB")
            x_original = TF.to_tensor(image).unsqueeze(0).to(device)
            original_size = x_original.shape[-2:]
            x = resize_if_needed(x_original, args.max_long)

            _, aux_c = conservative_model(x, return_all=True)
            _, aux_b = bright_model(x, return_all=True)

            gate_c = aux_c["gate"]
            gate_b = aux_b["gate"]
            noise_c = aux_c["noise"]
            noise_b = aux_b["noise"]
            y_base_c = aux_c["y_base"]
            y_base_b = aux_b["y_base"]

            # The two checkpoints share the same DIM-WRR backbone. Averaging keeps
            # the implementation robust if tiny serialization differences exist.
            y_base = 0.5 * (y_base_c + y_base_b)
            ybase_diff = (y_base_c - y_base_b).abs().mean()

            # Conservative expert: Release gate + medium risk suppression.
            under_c = make_under_map(x, threshold=0.35)
            over_c = make_over_map(y_base_c, threshold=0.88)
            suppress_c = (
                1.0
                - 0.20 * noise_c
                - 0.35 * over_c
                + 0.10 * under_c
            ).clamp(0.55, 1.00)
            e_c = gate_c * suppress_c

            # Bright expert: retrained gate + brightness-oriented risk control.
            under_b = make_under_map(x, threshold=0.45)
            over_b = make_over_map(y_base_b, threshold=0.90)
            suppress_b = (
                1.0
                - 0.08 * noise_b
                - 0.22 * over_b
                + 0.30 * under_b
            ).clamp(0.65, 1.20)
            e_b = gate_b * suppress_b

            # Image-level continuous router.
            router_under = make_under_map(x, threshold=args.router_under_th)
            router_over = make_over_map(y_base, threshold=args.router_over_th)
            under_mean = router_under.mean(dim=(1, 2, 3), keepdim=True)
            over_mean = router_over.mean(dim=(1, 2, 3), keepdim=True)
            noise_mean = 0.5 * (
                noise_c.mean(dim=(1, 2, 3), keepdim=True)
                + noise_b.mean(dim=(1, 2, 3), keepdim=True)
            )

            severity = torch.sigmoid(
                (under_mean - args.router_center)
                / max(args.router_width, 1e-6)
            )
            noise_excess = F.relu(noise_mean - 0.55)
            risk_safety = (
                1.0
                - args.risk_over_weight * over_mean
                - args.risk_noise_weight * noise_excess
            ).clamp(args.risk_floor, 1.0)
            rho = (
                args.rho_max * severity * risk_safety
            ).clamp(0.0, args.rho_max)

            effective_gate = (
                (1.0 - rho) * e_c + rho * e_b
            ).clamp(0.0, args.effective_max)
            output = (
                x + effective_gate * (y_base - x)
            ).clamp(0.0, 1.0)

            if output.shape[-2:] != original_size:
                output = F.interpolate(
                    output,
                    size=original_size,
                    mode="bilinear",
                    align_corners=False,
                ).clamp(0.0, 1.0)

            relative_path = image_path.relative_to(input_dir)
            save_tensor_image(output[0], output_dir / relative_path)

            stats["gate_c"].append(gate_c.mean().item())
            stats["gate_b"].append(gate_b.mean().item())
            stats["e_c"].append(e_c.mean().item())
            stats["e_b"].append(e_b.mean().item())
            stats["router_under"].append(under_mean.mean().item())
            stats["router_over"].append(over_mean.mean().item())
            stats["rho"].append(rho.mean().item())
            stats["effective"].append(effective_gate.mean().item())
            stats["ybase_diff"].append(ybase_diff.item())
            stats["input_luma"].append(image_luma(x_original).mean().item())
            stats["output_luma"].append(image_luma(output).mean().item())

            should_print = (
                index == 1
                or index % max(args.print_freq, 1) == 0
                or index == len(image_files)
            )
            if should_print:
                print(
                    "[{}/{}] under={:.4f} over={:.4f} rho={:.4f} "
                    "gate_c={:.4f} gate_b={:.4f} E_c={:.4f} "
                    "E_b={:.4f} E_final={:.4f} out_luma={:.4f}".format(
                        index,
                        len(image_files),
                        average(stats["router_under"]),
                        average(stats["router_over"]),
                        average(stats["rho"]),
                        average(stats["gate_c"]),
                        average(stats["gate_b"]),
                        average(stats["e_c"]),
                        average(stats["e_b"]),
                        average(stats["effective"]),
                        average(stats["output_luma"]),
                    ),
                    flush=True,
                )

    print("========== FINAL STATS ==========", flush=True)
    print(
        "gate conservative mean avg: {:.6f}".format(average(stats["gate_c"])),
        flush=True,
    )
    print(
        "gate bright mean avg: {:.6f}".format(average(stats["gate_b"])),
        flush=True,
    )
    print(
        "E conservative mean avg: {:.6f}".format(average(stats["e_c"])),
        flush=True,
    )
    print(
        "E bright mean avg: {:.6f}".format(average(stats["e_b"])),
        flush=True,
    )
    print(
        "router under mean avg: {:.6f}".format(
            average(stats["router_under"])
        ),
        flush=True,
    )
    print(
        "router over mean avg: {:.6f}".format(average(stats["router_over"])),
        flush=True,
    )
    print(
        "rho mean avg: {:.6f}".format(average(stats["rho"])),
        flush=True,
    )
    print(
        "effective gate mean avg: {:.6f}".format(
            average(stats["effective"])
        ),
        flush=True,
    )
    print(
        "y_base expert diff avg: {:.8f}".format(
            average(stats["ybase_diff"])
        ),
        flush=True,
    )
    print(
        "input luma mean avg: {:.6f}".format(average(stats["input_luma"])),
        flush=True,
    )
    print(
        "output luma mean avg: {:.6f}".format(average(stats["output_luma"])),
        flush=True,
    )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
