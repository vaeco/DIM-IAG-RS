"""Check final Release files and load both Dual-Gate Router checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dim_iag_rs.dim_wrr_iag import DIM_WRR_IAG, load_iag_checkpoint


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path, name: str, device: torch.device) -> None:
    print(f"========== {name} ==========")
    print("path:", path)
    print("size_bytes:", path.stat().st_size)
    print("md5:", md5sum(path))
    model = DIM_WRR_IAG(base_alpha=0.6, init_gate=0.45).to(device)
    load_iag_checkpoint(model, str(path), map_location=device, strict=False)
    print(f"[PASS] {name} checkpoint loaded")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conservative_ckpt",
        default="weights/dim_iag_rs_best.pth",
    )
    parser.add_argument(
        "--bright_ckpt",
        default="weights/dim_iag_rs_bright_gate_best.pth",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    conservative = Path(args.conservative_ckpt).expanduser().resolve()
    bright = Path(args.bright_ckpt).expanduser().resolve()

    missing = [str(path) for path in (conservative, bright) if not path.is_file()]
    if missing:
        print("[FAIL] Missing required checkpoint(s):")
        for path in missing:
            print(" -", path)
        raise SystemExit(1)

    load_checkpoint(conservative, "Conservative expert", device)
    load_checkpoint(bright, "Bright expert", device)
    print("========== RELEASE CHECK PASSED ==========")


if __name__ == "__main__":
    main()
