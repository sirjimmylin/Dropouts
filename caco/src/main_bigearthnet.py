from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms as T

from datasets.bigearthnet_dataset import Bigearthnet
from utils.downstream_utils import (
    build_feature_extractor,
    default_learning_rate,
    pick_device,
    seed_everything,
)


class BigEarthNetClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, in_features: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


def maybe_random_subset(dataset: Dataset, frac: float, seed: int) -> Dataset:
    if frac >= 1.0:
        return dataset
    if frac <= 0.0:
        raise ValueError("train_frac/val_frac must be > 0.")
    n = max(1, int(len(dataset) * frac))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=n, replace=False)
    return Subset(dataset, indices.tolist())


def compute_micro_map(all_targets: np.ndarray, all_probs: np.ndarray) -> float:
    try:
        return float(average_precision_score(all_targets, all_probs, average="micro") * 100.0)
    except ValueError:
        return float("nan")


def run_epoch(
    model: BigEarthNetClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    freeze_backbone: bool,
) -> tuple[float, float]:
    is_train = optimizer is not None
    if is_train:
        model.train()
        if freeze_backbone:
            model.backbone.eval()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0
    all_probs = []
    all_targets = []

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).float()

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, targets)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_targets.append(targets.detach().cpu().numpy())

        batch_size = targets.size(0)
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += int(batch_size)

    y_true = np.concatenate(all_targets, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    mAP = compute_micro_map(y_true, y_prob)
    mean_loss = total_loss / max(total_samples, 1)
    return mean_loss, mAP


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--backbone_type", choices=["random", "imagenet", "pretrain"], default="imagenet")
    parser.add_argument("--base_encoder", choices=["resnet18", "resnet34", "resnet50", "resnet101"], default="resnet18")
    parser.add_argument("--train_mode", choices=["linear", "finetune"], default="linear")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--milestones", type=int, nargs="*", default=[60, 80])
    parser.add_argument("--train_frac", type=float, default=1.0)
    parser.add_argument("--val_frac", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="results/bigearthnet")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("-d", "--description", type=str, default="")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = pick_device(args.device)

    transform = T.Compose([T.Resize((128, 128)), T.ToTensor()])
    train_dataset = Bigearthnet(root=args.data_dir, split="train", transform=transform)
    val_dataset = Bigearthnet(root=args.data_dir, split="val", transform=transform)
    train_dataset = maybe_random_subset(train_dataset, args.train_frac, args.seed)
    val_dataset = maybe_random_subset(val_dataset, args.val_frac, args.seed + 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    backbone, in_features = build_feature_extractor(
        backbone_type=args.backbone_type,
        base_encoder=args.base_encoder,
        ckpt_path=args.ckpt_path,
    )
    model = BigEarthNetClassifier(
        backbone=backbone,
        in_features=in_features,
        num_classes=19,
    ).to(device)

    freeze_backbone = args.train_mode == "linear"
    for param in model.backbone.parameters():
        param.requires_grad = not freeze_backbone

    learning_rate = args.learning_rate if args.learning_rate is not None else default_learning_rate(args.train_mode)
    trainable_params = model.classifier.parameters() if freeze_backbone else model.parameters()
    optimizer = optim.Adam(trainable_params, lr=learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
    criterion = nn.MultiLabelSoftMarginLoss()

    run_name = args.run_name
    if run_name is None:
        frac_tag = f"trainfrac{args.train_frac:g}"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{args.base_encoder}_{args.backbone_type}_{args.train_mode}_{frac_tag}_{stamp}"
        if args.description:
            run_name = f"{run_name}_{args.description}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics.csv"
    best_ckpt_path = run_dir / "best.pt"
    summary_path = run_dir / "summary.json"

    best_val_map = -1.0
    best_epoch = -1
    final_train_loss = None
    final_train_map = None
    final_val_loss = None
    final_val_map = None

    with metrics_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_map", "val_loss", "val_map", "lr"])
        for epoch in range(1, args.max_epochs + 1):
            train_loss, train_map = run_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                freeze_backbone=freeze_backbone,
            )
            val_loss, val_map = run_epoch(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                optimizer=None,
                device=device,
                freeze_backbone=freeze_backbone,
            )
            final_train_loss = float(train_loss)
            final_train_map = float(train_map)
            final_val_loss = float(val_loss)
            final_val_map = float(val_map)
            lr_now = optimizer.param_groups[0]["lr"]
            writer.writerow([epoch, train_loss, train_map, val_loss, val_map, lr_now])
            f.flush()

            print(
                f"[BigEarthNet][{epoch:03d}/{args.max_epochs}] "
                f"train_loss={train_loss:.4f} train_mAP={train_map:.2f} "
                f"val_loss={val_loss:.4f} val_mAP={val_map:.2f} lr={lr_now:.2e}"
            )

            if np.isfinite(val_map) and val_map > best_val_map:
                best_val_map = float(val_map)
                best_epoch = int(epoch)
                torch.save(
                    {
                        "epoch": epoch,
                        "best_val_map": best_val_map,
                        "args": vars(args),
                        "model_state_dict": model.state_dict(),
                    },
                    best_ckpt_path,
                )

            scheduler.step()

    best_ckpt_value = str(best_ckpt_path) if best_ckpt_path.exists() else None
    summary = {
        "dataset": "BigEarthNet",
        "backbone_type": args.backbone_type,
        "base_encoder": args.base_encoder,
        "train_mode": args.train_mode,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "best_val_map": best_val_map,
        "best_epoch": best_epoch,
        "final_train_loss": final_train_loss,
        "final_train_map": final_train_map,
        "final_val_loss": final_val_loss,
        "final_val_map": final_val_map,
        "num_train_samples": len(train_dataset),
        "num_val_samples": len(val_dataset),
        "metrics_csv": str(metrics_path),
        "best_checkpoint": best_ckpt_value,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved run artifacts to: {run_dir}")


if __name__ == "__main__":
    main()
