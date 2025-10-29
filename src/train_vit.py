import os, math, argparse, time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import timm
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import accuracy
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import CosineAnnealingLR


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='data/flowers')
    p.add_argument('--model', type=str, default='deit_small_patch16_224')
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--mixup', type=float, default=0.1)     # 0 pour désactiver
    p.add_argument('--cutmix', type=float, default=0.0)
    p.add_argument('--label_smoothing', type=float, default=0.1)
    p.add_argument('--val_split', type=float, default=0.2)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--out', type=str, default='checkpoints')
    return p.parse_args()


def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Transforms (RandAugment léger via torchvision, normalisation ImageNet)
    mean = (0.485, 0.456, 0.406); std = (0.229, 0.224, 0.225)
    train_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    full_ds = datasets.ImageFolder(args.data_dir, transform=train_tf)
    class_names = full_ds.classes
    num_classes = len(class_names)

    # Split train/val
    n_total = len(full_ds)
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    # val -> test transforms
    val_ds.dataset.transform = test_tf

    # pin_memory: True uniquement sur CUDA (pas sur MPS)
    pin = (device == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)

    # Modèle
    model = timm.create_model(args.model, pretrained=True, num_classes=num_classes)
    model.to(device)

    # Mixup/CutMix + label-smoothing
    if args.mixup > 0 or args.cutmix > 0:
        mixup_fn = Mixup(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
                         label_smoothing=args.label_smoothing, num_classes=num_classes)
        criterion = SoftTargetCrossEntropy()
    else:
        mixup_fn = None
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Optim + scheduler cosine (PyTorch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2)

    best_acc = 0.0
    Path(args.out).mkdir(parents=True, exist_ok=True)
    best_path = os.path.join(args.out, f'{args.model}_best.pth')

    def run_epoch(loader, train=True):
        model.train(train)
        total_loss, total_correct, total_samples = 0.0, 0, 0
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            if train:
                optimizer.zero_grad()
            if mixup_fn:
                images, targets = mixup_fn(images, targets)
            outputs = model(images)
            loss = criterion(outputs, targets)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            # accuracy top-1 (si SoftTarget, calcul sur labels "hard")
            with torch.no_grad():
                if isinstance(targets, torch.Tensor) and targets.ndim == 2:
                    hard = targets.argmax(dim=1)
                    acc1, = accuracy(outputs, hard, topk=(1,))
                else:
                    acc1, = accuracy(outputs, targets, topk=(1,))
            total_correct += acc1.item() * images.size(0) / 100.0
            total_samples += images.size(0)
        return total_loss / total_samples, total_correct / total_samples

    for epoch in range(args.epochs):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(train_loader, train=True)
        vl_loss, vl_acc = run_epoch(val_loader, train=False)
        dt = time.time() - t0
        print(f'Epoch {epoch+1:02d}/{args.epochs} - '
              f'train loss {tr_loss:.4f} acc {tr_acc:.3f} | '
              f'val loss {vl_loss:.4f} acc {vl_acc:.3f} [{dt:.1f}s]')

        # step du scheduler en fin d'epoch (pas par itération)
        scheduler.step()

        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save({'model': model.state_dict(), 'classes': class_names}, best_path)
            print(f'→ Saved best to {best_path} (val acc={best_acc:.3f})')

    # Rapport de classification sur la validation (avec le meilleur checkpoint)
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(1).cpu()
            y_pred.extend(preds.tolist())
            y_true.extend(targets.tolist())

    print(classification_report(y_true, y_pred, target_names=class_names))


if __name__ == "__main__":
    main()