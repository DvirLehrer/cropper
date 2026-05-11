#!/usr/bin/env python3

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageOps


@dataclass
class Sample:
    path: str
    label: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def list_image_files(root: str) -> List[str]:
    out: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                out.append(os.path.join(dirpath, fn))
    return out


def build_class_index(data_dir: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    class_names = []
    for name in os.listdir(data_dir):
        p = os.path.join(data_dir, name)
        if os.path.isdir(p):
            class_names.append(name)
    class_names.sort()
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    return class_to_idx, idx_to_class


def build_samples(data_dir: str, class_to_idx: Dict[str, int]) -> List[Sample]:
    samples: List[Sample] = []
    for cls_name, idx in class_to_idx.items():
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        files = list_image_files(cls_dir)
        for fp in files:
            samples.append(Sample(path=fp, label=idx))
    return samples


def split_train_val(samples: List[Sample], val_frac: float, seed: int) -> Tuple[List[Sample], List[Sample]]:
    # Simple stratified split per class.
    by_label: Dict[int, List[Sample]] = {}
    for s in samples:
        by_label.setdefault(s.label, []).append(s)

    rng = random.Random(seed)
    train: List[Sample] = []
    val: List[Sample] = []
    for label, items in by_label.items():
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_frac))) if len(items) >= 2 else 0
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _otsu_threshold_u8(arr_u8: np.ndarray) -> int:
    # Classic Otsu on 8-bit grayscale.
    hist = np.bincount(arr_u8.reshape(-1), minlength=256).astype(np.float64)
    total = arr_u8.size
    if total <= 0:
        return 127
    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / np.maximum(omega * (1.0 - omega), 1e-12)
    return int(np.argmax(sigma_b2))


def pil_to_tensor_gray(img: Image.Image, size: int, binarize: bool = True) -> "torch.Tensor":
    # Lazy import: torch requires running outside Cursor sandbox in some setups.
    import torch

    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)
    img = img.resize((size, size), resample=Image.BILINEAR)

    arr_u8 = np.asarray(img, dtype=np.uint8)
    # Normalize polarity: prefer dark text on light background.
    if int(np.median(arr_u8)) < 127:
        arr_u8 = 255 - arr_u8
    if binarize:
        thr = _otsu_threshold_u8(arr_u8)
        arr_u8 = np.where(arr_u8 >= thr, 255, 0).astype(np.uint8)

    arr = arr_u8.astype(np.float32) / 255.0  # (H,W)
    arr = (arr - 0.5) / 0.5  # normalize to ~[-1,1]
    t = torch.from_numpy(arr).unsqueeze(0)  # (1,H,W)
    return t


class CharDataset:
    def __init__(self, samples: List[Sample], image_size: int, augment: bool, seed: int):
        self.samples = samples
        self.image_size = image_size
        self.augment = augment
        self.rng = random.Random(seed)
        self.binarize = True

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, img: Image.Image) -> Image.Image:
        # Lightweight augmentation without torchvision:
        # small rotations + optional invert (some scans differ).
        angle = self.rng.uniform(-5.0, 5.0)
        img = img.rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor=255)
        # Center-crop back to square-ish before resize
        w, h = img.size
        m = min(w, h)
        left = (w - m) // 2
        top = (h - m) // 2
        img = img.crop((left, top, left + m, top + m))
        if self.rng.random() < 0.10:
            img = ImageOps.invert(ImageOps.grayscale(img)).convert("RGB")
        return img

    def __getitem__(self, idx: int):
        import torch

        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        if self.augment:
            img = self._augment(img)
        x = pil_to_tensor_gray(img, self.image_size, binarize=self.binarize)
        y = torch.tensor(s.label, dtype=torch.long)
        return x, y


def make_loader(dataset: CharDataset, batch_size: int, shuffle: bool):
    import torch
    from torch.utils.data import DataLoader

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)


def build_model(num_classes: int):
    import torch.nn as nn

    # Small CNN for 32x32 grayscale.
    return nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),  # /2
        nn.Dropout(0.10),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),  # /4
        nn.Dropout(0.15),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(128, num_classes),
    )


def accuracy(logits, y) -> float:
    import torch

    preds = torch.argmax(logits, dim=1)
    return float((preds == y).float().mean().item())

def confusion_matrix(num_classes: int, y_true: List[int], y_pred: List[int]) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def main():
    parser = argparse.ArgumentParser(description="Train a simple CNN on the exported char dataset.")
    parser.add_argument("--data-dir", default="test_output/char_dataset", help="Dataset root with per-class subfolders.")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out-dir", default="test_output/cnn_model", help="Where to write model + metadata.")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience (epochs without improvement).")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Minimum improvement to reset patience.")
    parser.add_argument("--no-class-weights", action="store_true", help="Disable class-weighted loss.")
    parser.add_argument(
        "--binarize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Binarize crops (Otsu) after autocontrast; helps match OCR-style glyph shapes.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    class_to_idx, idx_to_class = build_class_index(args.data_dir)
    samples = build_samples(args.data_dir, class_to_idx)
    if not samples:
        raise SystemExit(f"No samples found under: {args.data_dir}")

    train_s, val_s = split_train_val(samples, val_frac=args.val_frac, seed=args.seed)
    print(f"Classes: {len(class_to_idx)}")
    print(f"Samples: total={len(samples)} train={len(train_s)} val={len(val_s)}")

    import torch
    import torch.nn as nn
    import torch.optim as optim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = CharDataset(train_s, image_size=args.image_size, augment=True, seed=args.seed)
    val_ds = CharDataset(val_s, image_size=args.image_size, augment=False, seed=args.seed)
    train_ds.binarize = bool(args.binarize)
    val_ds.binarize = bool(args.binarize)
    train_loader = make_loader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = make_loader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = build_model(num_classes=len(class_to_idx)).to(device)
    # Class-weighted loss helps with imbalance.
    if args.no_class_weights:
        criterion = nn.CrossEntropyLoss()
    else:
        counts = np.zeros((len(class_to_idx),), dtype=np.int64)
        for s in train_s:
            counts[s.label] += 1
        # Weight inversely proportional to frequency (smoothed).
        weights = 1.0 / np.maximum(counts.astype(np.float32), 1.0)
        weights = weights / weights.mean()
        w = torch.tensor(weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_left = int(args.patience)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        train_accs = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
            train_accs.append(accuracy(logits.detach(), y))

        model.eval()
        val_losses = []
        val_accs = []
        y_true: List[int] = []
        y_pred: List[int] = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_losses.append(float(loss.item()))
                val_accs.append(accuracy(logits, y))
                preds = torch.argmax(logits, dim=1)
                y_true.extend(y.cpu().numpy().tolist())
                y_pred.extend(preds.cpu().numpy().tolist())

        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        val_acc = float(np.mean(val_accs)) if val_accs else 0.0
        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
        train_acc = float(np.mean(train_accs)) if train_accs else 0.0
        lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step(val_loss)

        print(
            f"epoch {epoch:02d} | "
            f"lr={lr:.2e} | "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f}"
        )

        # Save best checkpoint + evaluation artifacts
        improved = (best_val_loss - val_loss) > float(args.min_delta)
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_left = int(args.patience)

            os.makedirs(args.out_dir, exist_ok=True)
            model_path = os.path.join(args.out_dir, "model.pt")
            meta_path = os.path.join(args.out_dir, "meta.json")
            cm_path = os.path.join(args.out_dir, "confusion_matrix.csv")
            per_class_path = os.path.join(args.out_dir, "per_class_accuracy.json")

            torch.save(model.state_dict(), model_path)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "data_dir": os.path.abspath(args.data_dir),
                        "image_size": args.image_size,
                        "class_to_idx": class_to_idx,
                        "idx_to_class": idx_to_class,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            cm = confusion_matrix(len(class_to_idx), y_true, y_pred)
            np.savetxt(cm_path, cm, delimiter=",", fmt="%d")

            per_class = {}
            for i in range(len(class_to_idx)):
                denom = int(cm[i, :].sum())
                per_class[idx_to_class[i]] = float(cm[i, i] / denom) if denom > 0 else None
            with open(per_class_path, "w", encoding="utf-8") as f:
                json.dump(per_class, f, ensure_ascii=False, indent=2)

            print(f"  ↳ new best @ epoch {epoch}: val_loss={val_loss:.4f} val_acc={val_acc:.4f} (saved)")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping: best epoch={best_epoch} best val_loss={best_val_loss:.4f}")
                break

    print(f"Best epoch={best_epoch} best val_loss={best_val_loss:.4f} (model in {args.out_dir})")


if __name__ == "__main__":
    main()


