# Add repository root to PYTHONPATH when running this script directly.
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dim_iag_rs.dim_wrr_iag import DIM_WRR_IAG, load_iag_checkpoint


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".PNG", ".JPG", ".JPEG", ".BMP")


PRESETS = {
    # C1: closest to Version A, mild suppression
    "mild": {
        "w_noise": 0.10,
        "w_over": 0.20,
        "w_under": 0.05,
        "s_min": 0.70,
        "s_max": 1.05,
        "over_th": 0.90,
        "under_th": 0.35,
    },
    # C2: more conservative, recommended main test
    "medium": {
        "w_noise": 0.20,
        "w_over": 0.35,
        "w_under": 0.10,
        "s_min": 0.55,
        "s_max": 1.00,
        "over_th": 0.88,
        "under_th": 0.35,
    },
    # C3: strong residual suppression, closer to raw distribution
    "strong": {
        "w_noise": 0.35,
        "w_over": 0.50,
        "w_under": 0.15,
        "s_min": 0.35,
        "s_max": 0.95,
        "over_th": 0.85,
        "under_th": 0.40,
    },
}


def list_images(input_dir):
    input_dir = Path(input_dir)
    files = []
    for ext in IMG_EXTS:
        files.extend(input_dir.rglob(f"*{ext}"))
    return sorted(list(set(files)))


def load_rgb(path):
    return Image.open(path).convert("RGB")


def save_tensor_img(t, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = TF.to_pil_image(t.detach().cpu().clamp(0, 1))
    img.save(path)


def luma_fn(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def normalize_over_risk(y_base, over_th):
    """
    over-exposure risk from enhanced base image.
    Larger when Y_base is close to saturation.
    """
    over = F.relu(y_base - over_th).mean(dim=1, keepdim=True)
    over = over / max(1e-6, (1.0 - over_th))
    return over.clamp(0, 1)


def normalize_under_map(x, under_th):
    """
    under-exposure map from raw image.
    Larger in dark regions, allowing slightly more residual.
    """
    luma = luma_fn(x)
    under = F.relu(under_th - luma) / max(1e-6, under_th)
    return under.clamp(0, 1)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ckpt", required=True)

    parser.add_argument("--base_alpha", type=float, default=0.6)
    parser.add_argument("--init_gate", type=float, default=0.45)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_long", type=int, default=0)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--print_freq", type=int, default=100)

    parser.add_argument("--preset", choices=["mild", "medium", "strong", "custom"], default="medium")

    # used only when preset=custom, otherwise preset values override these
    parser.add_argument("--w_noise", type=float, default=0.20)
    parser.add_argument("--w_over", type=float, default=0.35)
    parser.add_argument("--w_under", type=float, default=0.10)
    parser.add_argument("--s_min", type=float, default=0.55)
    parser.add_argument("--s_max", type=float, default=1.00)
    parser.add_argument("--over_th", type=float, default=0.88)
    parser.add_argument("--under_th", type=float, default=0.35)

    parser.add_argument("--save_maps", action="store_true")
    parser.add_argument("--maps_dir", default="")

    return parser.parse_args()


def main():
    args = get_args()

    if args.preset != "custom":
        cfg = PRESETS[args.preset]
        args.w_noise = cfg["w_noise"]
        args.w_over = cfg["w_over"]
        args.w_under = cfg["w_under"]
        args.s_min = cfg["s_min"]
        args.s_max = cfg["s_max"]
        args.over_th = cfg["over_th"]
        args.under_th = cfg["under_th"]

    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu explicitly.")
    if not Path(args.ckpt).is_file():
        raise FileNotFoundError(f"Missing trained single-gate checkpoint: {args.ckpt}")
    device = torch.device(requested_device)

    model = DIM_WRR_IAG(base_alpha=args.base_alpha, init_gate=args.init_gate).to(device).eval()
    load_iag_checkpoint(model, args.ckpt, map_location=device)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    files = list_images(input_dir)
    if not files:
        raise ValueError(f"No supported images found in {input_dir}")
    if args.print_freq < 1:
        raise ValueError("--print_freq must be positive")
    if args.max_images and args.max_images > 0:
        files = files[: args.max_images]

    print("==== enhance config ====")
    print("method: DIM-IAG-RS")
    print("input_dir:", input_dir)
    print("output_dir:", output_dir)
    print("ckpt:", args.ckpt)
    print("device:", device)
    print("num_images:", len(files))
    print("base_alpha:", args.base_alpha)
    print("preset:", args.preset)
    print("w_noise:", args.w_noise)
    print("w_over:", args.w_over)
    print("w_under:", args.w_under)
    print("s_min:", args.s_min)
    print("s_max:", args.s_max)
    print("over_th:", args.over_th)
    print("under_th:", args.under_th)
    print("NOTE: no training, no downstream labels, inference-only residual suppression.")

    gate_means, gate_mins, gate_maxs = [], [], []
    noise_means, over_means, under_means = [], [], []
    suppress_means, suppress_mins, suppress_maxs = [], [], []
    eff_gate_means = []

    maps_dir = Path(args.maps_dir) if args.maps_dir else output_dir.parent / (output_dir.name + "_maps")

    with torch.no_grad():
        for idx, p in enumerate(files, 1):
            img = load_rgb(p)

            if args.max_long and max(img.size) > args.max_long:
                w, h = img.size
                scale = args.max_long / max(w, h)
                nw, nh = int(round(w * scale)), int(round(h * scale))
                img = img.resize((nw, nh), Image.BICUBIC)

            x = TF.to_tensor(img).unsqueeze(0).to(device)

            # Version A forward, reuse learned maps
            y_a, aux = model(x, return_all=True)

            gate = aux["gate"]
            noise = aux["noise"]
            y_base = aux.get("y_base", None)
            if y_base is None:
                raise RuntimeError("aux['y_base'] not found. Please check DIM_WRR_IAG return_all output.")

            over = normalize_over_risk(y_base, args.over_th)
            under = normalize_under_map(x, args.under_th)

            # Risk-aware residual suppression map
            suppress = 1.0 - args.w_noise * noise - args.w_over * over + args.w_under * under
            suppress = suppress.clamp(args.s_min, args.s_max)

            residual = y_base - x
            effective_gate = gate * suppress
            y_c = torch.clamp(x + effective_gate * residual, 0.0, 1.0)

            rel = p.relative_to(input_dir)
            out_path = output_dir / rel
            save_tensor_img(y_c[0], out_path)

            if args.save_maps:
                save_tensor_img(gate[0].repeat(3, 1, 1), maps_dir / "gate" / rel)
                save_tensor_img(noise[0].repeat(3, 1, 1), maps_dir / "noise" / rel)
                save_tensor_img(over[0].repeat(3, 1, 1), maps_dir / "over" / rel)
                save_tensor_img(under[0].repeat(3, 1, 1), maps_dir / "under" / rel)
                save_tensor_img(suppress[0].repeat(3, 1, 1), maps_dir / "suppress" / rel)
                save_tensor_img(effective_gate[0].repeat(3, 1, 1), maps_dir / "effective_gate" / rel)

            gate_means.append(gate.mean().item())
            gate_mins.append(gate.min().item())
            gate_maxs.append(gate.max().item())

            noise_means.append(noise.mean().item())
            over_means.append(over.mean().item())
            under_means.append(under.mean().item())

            suppress_means.append(suppress.mean().item())
            suppress_mins.append(suppress.min().item())
            suppress_maxs.append(suppress.max().item())
            eff_gate_means.append(effective_gate.mean().item())

            if idx % args.print_freq == 0 or idx == len(files):
                print(
                    f"[{idx}/{len(files)}] "
                    f"gate={sum(gate_means)/len(gate_means):.4f} "
                    f"noise={sum(noise_means)/len(noise_means):.4f} "
                    f"over={sum(over_means)/len(over_means):.4f} "
                    f"under={sum(under_means)/len(under_means):.4f} "
                    f"suppress mean/min/max="
                    f"{sum(suppress_means)/len(suppress_means):.4f}/"
                    f"{sum(suppress_mins)/len(suppress_mins):.4f}/"
                    f"{sum(suppress_maxs)/len(suppress_maxs):.4f} "
                    f"effective_gate={sum(eff_gate_means)/len(eff_gate_means):.4f}"
                )

    print("==== FINAL STATS ====")
    print(f"gate mean avg: {sum(gate_means)/len(gate_means):.6f}")
    print(f"gate min avg: {sum(gate_mins)/len(gate_mins):.6f}")
    print(f"gate max avg: {sum(gate_maxs)/len(gate_maxs):.6f}")
    print(f"noise mean avg: {sum(noise_means)/len(noise_means):.6f}")
    print(f"over mean avg: {sum(over_means)/len(over_means):.6f}")
    print(f"under mean avg: {sum(under_means)/len(under_means):.6f}")
    print(f"suppress mean avg: {sum(suppress_means)/len(suppress_means):.6f}")
    print(f"suppress min avg: {sum(suppress_mins)/len(suppress_mins):.6f}")
    print(f"suppress max avg: {sum(suppress_maxs)/len(suppress_maxs):.6f}")
    print(f"effective gate mean avg: {sum(eff_gate_means)/len(eff_gate_means):.6f}")
    print("Done.")


if __name__ == "__main__":
    main()
