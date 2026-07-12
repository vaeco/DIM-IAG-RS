import argparse
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


OFFICIAL_CLASSES = [
    'Bicycle',
    'Car',
    'Motorbike',
    'Bus',
    'Boat',
    'Cat',
    'Dog',
    'Bottle',
    'Cup',
    'Chair',
]


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.JPEG'}


class CODaNFolderOfficial(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        for label, cls in enumerate(OFFICIAL_CLASSES):
            cls_dir = self.root / cls
            if not cls_dir.exists():
                raise FileNotFoundError(f'class dir not found: {cls_dir}')

            imgs = sorted([
                p for p in cls_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
            ])

            for p in imgs:
                self.samples.append((p, label, cls))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, label, cls = self.samples[idx]
        img = Image.open(p).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def clean_state_dict(state):
    new_state = {}
    for k, v in state.items():
        nk = k
        for prefix in ['module.', 'model.']:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        new_state[nk] = v
    return new_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    ckpt_path = Path(args.ckpt)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = CODaNFolderOfficial(data_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print('data_dir:', data_dir)
    print('num images:', len(dataset))
    print('official classes:', OFFICIAL_CLASSES)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(OFFICIAL_CLASSES))
    model = model.to(device)

    ckpt = torch.load(str(ckpt_path), map_location=device)

    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            state = ckpt['state_dict']
        elif 'model' in ckpt:
            state = ckpt['model']
        else:
            state = ckpt
    else:
        state = ckpt

    state = clean_state_dict(state)
    missing, unexpected = model.load_state_dict(state, strict=False)

    print('loaded checkpoint:', ckpt_path)
    print('missing keys:', len(missing))
    print('unexpected keys:', len(unexpected))
    if missing:
        print('first missing:', missing[:10])
    if unexpected:
        print('first unexpected:', unexpected[:10])

    model.eval()

    correct1 = 0
    correct5 = 0
    total = 0

    class_correct = torch.zeros(len(OFFICIAL_CLASSES), dtype=torch.long)
    class_total = torch.zeros(len(OFFICIAL_CLASSES), dtype=torch.long)

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            logits = model(imgs)

            _, pred1 = logits.max(1)
            correct1 += pred1.eq(targets).sum().item()

            _, pred5 = logits.topk(min(5, len(OFFICIAL_CLASSES)), dim=1)
            correct5 += pred5.eq(targets.view(-1, 1)).sum().item()

            total += targets.size(0)

            for t, p in zip(targets.cpu(), pred1.cpu()):
                class_total[t] += 1
                if t == p:
                    class_correct[t] += 1

    top1 = correct1 / total * 100
    top5 = correct5 / total * 100

    print()
    print('========== CODaN Classification Result ==========')
    print('Dataset:', data_dir.name)
    print('Top-1 Acc:', round(top1, 2))
    print('Top-5 Acc:', round(top5, 2))
    print('Total:', total)
    print('=================================================')

    print()
    print('Per-class Top-1 Acc:')
    for i, name in enumerate(OFFICIAL_CLASSES):
        acc = class_correct[i].item() / max(class_total[i].item(), 1) * 100
        print(f'{name:12s}: {acc:.2f}%  ({class_correct[i].item()}/{class_total[i].item()})')


if __name__ == '__main__':
    main()
