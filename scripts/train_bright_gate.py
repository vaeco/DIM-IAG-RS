# Add repository root to PYTHONPATH when running this script directly.
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from dim_iag_rs.dim_wrr_iag import DIM_WRR_IAG, load_iag_checkpoint, load_wrr_checkpoint_into_iag

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".PNG", ".JPG", ".JPEG", ".BMP")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images(folder):
    folder = Path(folder)
    files = []
    for ext in IMG_EXTS:
        files.extend(folder.glob(f"*{ext}"))
    return sorted(files)


def build_pairs(input_dir, target_dir):
    input_dir = Path(input_dir)
    target_dir = Path(target_dir)
    inputs = list_images(input_dir)
    targets = list_images(target_dir)
    target_map = {p.stem: p for p in targets}
    pairs = []
    for p in inputs:
        if p.stem in target_map:
            pairs.append((p, target_map[p.stem]))
    if not pairs:
        raise RuntimeError(f"No paired images found by same stem: {input_dir} vs {target_dir}")
    return pairs


def pil_to_tensor(img):
    arr = np.asarray(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    arr = arr[:, :, :3]
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


class PairedDataset(Dataset):
    def __init__(self, pairs, crop_size=384, train=True):
        self.pairs = pairs
        self.crop_size = int(crop_size)
        self.train = bool(train)

    def __len__(self):
        return len(self.pairs)

    def _resize_if_needed(self, x_img, y_img):
        w, h = x_img.size
        cs = self.crop_size
        if w < cs or h < cs:
            nw, nh = max(w, cs), max(h, cs)
            x_img = x_img.resize((nw, nh), Image.BICUBIC)
            y_img = y_img.resize((nw, nh), Image.BICUBIC)
        return x_img, y_img

    def _crop(self, x_img, y_img):
        w, h = x_img.size
        cs = self.crop_size
        if w == cs and h == cs:
            return x_img, y_img
        if self.train:
            left = random.randint(0, w - cs)
            top = random.randint(0, h - cs)
        else:
            left = max((w - cs) // 2, 0)
            top = max((h - cs) // 2, 0)
        return x_img.crop((left, top, left + cs, top + cs)), y_img.crop((left, top, left + cs, top + cs))

    def __getitem__(self, idx):
        x_path, y_path = self.pairs[idx]
        x_img = Image.open(x_path).convert("RGB")
        y_img = Image.open(y_path).convert("RGB")
        if x_img.size != y_img.size:
            y_img = y_img.resize(x_img.size, Image.BICUBIC)
        x_img, y_img = self._resize_if_needed(x_img, y_img)
        x_img, y_img = self._crop(x_img, y_img)
        if self.train and random.random() < 0.5:
            x_img = x_img.transpose(Image.FLIP_LEFT_RIGHT)
            y_img = y_img.transpose(Image.FLIP_LEFT_RIGHT)
        return pil_to_tensor(x_img), pil_to_tensor(y_img), x_path.name


def luma(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def normalize_over_risk(y_base, over_th):
    over = F.relu(y_base - over_th).mean(dim=1, keepdim=True)
    return (over / max(1e-6, 1.0 - over_th)).clamp(0, 1)


def normalize_under_map(x, under_th):
    under = F.relu(under_th - luma(x)) / max(1e-6, under_th)
    return under.clamp(0, 1)


def charbonnier_loss(x, y, eps=1e-3):
    return torch.mean(torch.sqrt((x - y) ** 2 + eps ** 2))


def ssim_loss(x, y):
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2) + 1e-8
    )
    return 1.0 - ssim.mean()


def sobel_edge(x_luma):
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x_luma.dtype, device=x_luma.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x_luma.dtype, device=x_luma.device).view(1, 1, 3, 3)
    gx = F.conv2d(x_luma, kx, padding=1)
    gy = F.conv2d(x_luma, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6).clamp(0, 1)


def edge_loss(pred, target):
    return F.l1_loss(sobel_edge(luma(pred)), sobel_edge(luma(target)))


def exposure_loss(pred, target, patch=16):
    return F.l1_loss(F.avg_pool2d(pred, patch, patch), F.avg_pool2d(target, patch, patch))


def tv_loss(x):
    return torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])) + torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))


def masked_l1(a, b, mask, eps=1e-6):
    return (torch.abs(a - b) * mask).sum() / (mask.sum() + eps)


def make_final_output(model, x, args):
    # model.forward already gives y = x + gate * residual, but here we train the same RS/boost fusion used for final inference.
    _, aux = model(x, return_all=True)
    gate = aux["gate"]
    noise = aux["noise"]
    y_base = aux["y_base"]

    over = normalize_over_risk(y_base, args.over_th)
    under = normalize_under_map(x, args.under_th)

    # Brightness-aware RS: keep suppression for noise/overexposure, but allow boost in dark useful regions.
    suppress = 1.0 - args.w_noise * noise - args.w_over * over + args.w_under * under
    suppress = suppress.clamp(args.s_min, args.s_max)

    residual = y_base - x
    effective_gate = gate * suppress
    y = torch.clamp(x + effective_gate * residual, 0.0, 1.0)

    aux.update({
        "over": over,
        "under": under,
        "suppress": suppress,
        "effective_gate": effective_gate,
        "final": y,
    })
    return y, aux


def residual_coeff_target(x, y_base, target, e_max=1.20, min_residual=0.025):
    residual = y_base - x
    target_delta = target - x
    numerator = (target_delta * residual).sum(dim=1, keepdim=True)
    denominator = (residual * residual).sum(dim=1, keepdim=True) + 1e-6
    coeff = (numerator / denominator).clamp(0.0, e_max)
    valid = (residual.abs().mean(dim=1, keepdim=True) > min_residual).float()
    return coeff, valid


def compute_loss(x, target, y, aux, args):
    rec = charbonnier_loss(y, target)
    ssim = ssim_loss(y, target)
    edge = edge_loss(y, target)
    exp = exposure_loss(y, target)

    lx = luma(x)
    lt = luma(target)
    ly = luma(y)

    # Brightness-needed area: raw is dark and GT is clearly brighter.
    dark_need = ((lx < args.dark_th) & ((lt - lx) > args.need_delta)).float()
    dark_luma = masked_l1(ly, lt, dark_need)

    coeff_t, coeff_valid = residual_coeff_target(x, aux["y_base"], target, e_max=args.s_max)
    coeff_mask = coeff_valid * dark_need
    coeff = masked_l1(aux["effective_gate"], coeff_t, coeff_mask)

    # Avoid a trivial all-bright solution and keep maps spatially stable.
    over_penalty = torch.mean(F.relu(ly - args.over_luma_penalty) ** 2)
    tv_gate = tv_loss(aux["gate"])
    tv_eff = tv_loss(aux["effective_gate"])

    loss = (
        rec
        + args.w_ssim_loss * ssim
        + args.w_edge_loss * edge
        + args.w_exp_loss * exp
        + args.w_dark_loss * dark_luma
        + args.w_coeff_loss * coeff
        + args.w_over_loss * over_penalty
        + args.w_tv_gate * tv_gate
        + args.w_tv_eff * tv_eff
    )
    parts = {
        "loss": loss.detach(),
        "rec": rec.detach(),
        "ssim": ssim.detach(),
        "edge": edge.detach(),
        "exp": exp.detach(),
        "dark": dark_luma.detach(),
        "coeff": coeff.detach(),
        "overp": over_penalty.detach(),
        "tv_g": tv_gate.detach(),
        "tv_e": tv_eff.detach(),
        "gate_mean": aux["gate"].mean().detach(),
        "suppress_mean": aux["suppress"].mean().detach(),
        "eff_mean": aux["effective_gate"].mean().detach(),
        "under_mean": aux["under"].mean().detach(),
        "noise_mean": aux["noise"].mean().detach(),
    }
    return loss, parts


@torch.no_grad()
def validate(model, loader, device, args):
    model.eval()
    sums = {}
    n = 0
    for x, target, _ in loader:
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        y, aux = make_final_output(model, x, args)
        loss, parts = compute_loss(x, target, y, aux, args)
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + float(v.cpu())
        n += 1
    return {k: v / max(n, 1) for k, v in sums.items()}


def set_trainable(model, args):
    for p in model.parameters():
        p.requires_grad_(False)

    train_modules = []
    # Default: train gate only. This is the safest first run.
    train_modules.append(model.gate)

    if args.train_illum:
        train_modules.append(model.illumination)
    if args.train_noise:
        train_modules.append(model.noise)
    if args.unfreeze_refiner:
        for name, module in model.base.named_modules():
            if "refiner" in name.lower():
                train_modules.append(module)

    for module in train_modules:
        for p in module.parameters():
            p.requires_grad_(True)

    param_groups = []
    gate_params = [p for p in model.gate.parameters() if p.requires_grad]
    if gate_params:
        param_groups.append({"params": gate_params, "lr": args.lr_gate, "name": "gate"})
    illum_params = [p for p in model.illumination.parameters() if p.requires_grad]
    if illum_params:
        param_groups.append({"params": illum_params, "lr": args.lr_iag, "name": "illumination"})
    noise_params = [p for p in model.noise.parameters() if p.requires_grad]
    if noise_params:
        param_groups.append({"params": noise_params, "lr": args.lr_iag, "name": "noise"})
    refiner_params = []
    if args.unfreeze_refiner:
        for name, module in model.base.named_modules():
            if "refiner" in name.lower():
                refiner_params += [p for p in module.parameters() if p.requires_grad]
        if refiner_params:
            param_groups.append({"params": refiner_params, "lr": args.lr_refiner, "name": "refiner"})

    return param_groups


def save_ckpt(path, model, optimizer, epoch, best_val, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "best_val": best_val,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }, str(path))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--target_dir", required=True)
    parser.add_argument("--ckpt", default="weights/dim_iag_rs_best.pth")
    parser.add_argument("--load_mode", choices=["iag", "wrr"], default="iag")
    parser.add_argument("--save_dir", default="weights/bright_gate_training")

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=384)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--max_train_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--print_freq", type=int, default=20)

    parser.add_argument("--base_alpha", type=float, default=0.6)
    parser.add_argument("--init_gate", type=float, default=0.55)
    parser.add_argument("--gate_hidden", type=int, default=24)

    # Trainable scope. First run: gate only. Second run can add --train_illum --train_noise.
    parser.add_argument("--train_illum", action="store_true")
    parser.add_argument("--train_noise", action="store_true")
    parser.add_argument("--unfreeze_refiner", action="store_true")
    parser.add_argument("--lr_gate", type=float, default=1e-4)
    parser.add_argument("--lr_iag", type=float, default=3e-5)
    parser.add_argument("--lr_refiner", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # RS/boost fusion used during training. s_max > 1 lets dark useful regions exceed y_base slightly.
    parser.add_argument("--w_noise", type=float, default=0.08)
    parser.add_argument("--w_over", type=float, default=0.22)
    parser.add_argument("--w_under", type=float, default=0.30)
    parser.add_argument("--s_min", type=float, default=0.65)
    parser.add_argument("--s_max", type=float, default=1.20)
    parser.add_argument("--over_th", type=float, default=0.90)
    parser.add_argument("--under_th", type=float, default=0.45)

    # Loss weights.
    parser.add_argument("--w_ssim_loss", type=float, default=0.15)
    parser.add_argument("--w_edge_loss", type=float, default=0.05)
    parser.add_argument("--w_exp_loss", type=float, default=0.10)
    parser.add_argument("--w_dark_loss", type=float, default=0.25)
    parser.add_argument("--w_coeff_loss", type=float, default=0.08)
    parser.add_argument("--w_over_loss", type=float, default=0.02)
    parser.add_argument("--w_tv_gate", type=float, default=0.001)
    parser.add_argument("--w_tv_eff", type=float, default=0.001)
    parser.add_argument("--dark_th", type=float, default=0.50)
    parser.add_argument("--need_delta", type=float, default=0.04)
    parser.add_argument("--over_luma_penalty", type=float, default=0.93)
    return parser.parse_args()


def main():
    args = get_args()
    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pairs = build_pairs(args.input_dir, args.target_dir)
    random.shuffle(pairs)
    val_n = max(1, int(len(pairs) * args.val_ratio))
    val_pairs = pairs[:val_n]
    train_pairs = pairs[val_n:]
    print("total pairs:", len(pairs))
    print("train pairs:", len(train_pairs))
    print("val pairs:", len(val_pairs))
    print("device:", device)

    train_set = PairedDataset(train_pairs, crop_size=args.crop_size, train=True)
    val_set = PairedDataset(val_pairs, crop_size=args.crop_size, train=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)

    model = DIM_WRR_IAG(base_alpha=args.base_alpha, init_gate=args.init_gate, gate_hidden=args.gate_hidden).to(device)
    if args.load_mode == "iag":
        model = load_iag_checkpoint(model, args.ckpt, map_location=device, strict=False)
    else:
        model = load_wrr_checkpoint_into_iag(model, args.ckpt, map_location=device)

    param_groups = set_trainable(model, args)
    if not param_groups:
        raise RuntimeError("No trainable parameters. Check train flags.")
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)

    print("==== trainable params ====")
    for group in param_groups:
        print(group.get("name", "group"), sum(p.numel() for p in group["params"]), "lr", group["lr"])
    print("==== fusion config ====")
    print("w_noise/w_over/w_under:", args.w_noise, args.w_over, args.w_under)
    print("s_min/s_max:", args.s_min, args.s_max)
    print("over_th/under_th:", args.over_th, args.under_th)

    best_val = 1e9
    for epoch in range(1, args.epochs + 1):
        model.train()
        # Keep frozen base in eval mode, so only selected IAG modules update.
        model.base.eval()
        if not args.train_illum:
            model.illumination.eval()
        if not args.train_noise:
            model.noise.eval()

        running = {}
        steps = 0
        for step, (x, target, names) in enumerate(train_loader, 1):
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=args.amp):
                y, aux = make_final_output(model, x, args)
                loss, parts = compute_loss(x, target, y, aux, args)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_([p for g in param_groups for p in g["params"]], max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            steps += 1
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + float(v.cpu())

            if step == 1 or step % args.print_freq == 0:
                avg = {k: v / steps for k, v in running.items()}
                print(
                    f"epoch {epoch:03d} step {step:04d}/{len(train_loader)} "
                    f"loss={avg['loss']:.5f} rec={avg['rec']:.5f} ssim={avg['ssim']:.5f} "
                    f"exp={avg['exp']:.5f} dark={avg['dark']:.5f} coeff={avg['coeff']:.5f} "
                    f"gate={avg['gate_mean']:.4f} S={avg['suppress_mean']:.4f} E={avg['eff_mean']:.4f} "
                    f"under={avg['under_mean']:.4f} noise={avg['noise_mean']:.4f}"
                )

            if args.max_train_steps > 0 and step >= args.max_train_steps:
                break

        val = validate(model, val_loader, device, args)
        print(
            f"epoch {epoch:03d} VAL loss={val['loss']:.5f} rec={val['rec']:.5f} "
            f"ssim={val['ssim']:.5f} exp={val['exp']:.5f} dark={val['dark']:.5f} coeff={val['coeff']:.5f} "
            f"gate={val['gate_mean']:.4f} S={val['suppress_mean']:.4f} E={val['eff_mean']:.4f}"
        )

        latest = Path(args.save_dir) / "latest.pth"
        save_ckpt(latest, model, optimizer, epoch, best_val, args)
        if val["loss"] < best_val:
            best_val = val["loss"]
            best = Path(args.save_dir) / "best.pth"
            save_ckpt(best, model, optimizer, epoch, best_val, args)
            print("saved best:", best, "best_val:", best_val)

    print("training done. best_val:", best_val)


if __name__ == "__main__":
    main()
