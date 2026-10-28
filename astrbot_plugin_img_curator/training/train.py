r"""Train DINOv2 ViT-S/14 for collect/skip classification on Windows + CUDA.

Example:
  python training/train.py --data-root D:\data\taste --output-dir D:\runs\taste_v1 ^
      --epochs 30 --batch-size 32 --unfreeze-last-blocks 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import (  # noqa: E402
    TasteDataset,
    class_counts,
    class_weights,
    dedupe_records,
    discover_records,
    make_loader,
    write_dedupe_report,
)
from model import (  # noqa: E402
    BACKBONE_NAME,
    CLASS_NAMES,
    IMAGE_SIZE,
    build_model,
    eval_transform,
    set_trainable,
    train_transform,
    write_model_config,
)
from utils import binary_metrics, resolve_device, save_json, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DINOv2 collect/skip fine-tuning")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backbone", default=BACKBONE_NAME)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--unfreeze-last-blocks", type=int, default=0)
    parser.add_argument("--full-finetune", action="store_true")
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--dedupe-threshold", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--version", default="1.0.0")
    return parser.parse_args()


def evaluate_loss(model, loader, criterion, device, amp_enabled: bool) -> tuple[float, list[int], list[float]]:
    model.eval()
    losses = []
    labels: list[int] = []
    scores: list[float] = []
    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets)
            losses.append(float(loss.detach().cpu()))
            probs = torch.softmax(logits.float(), dim=1)[:, 1]
            labels.extend(int(x) for x in targets.cpu().tolist())
            scores.extend(float(x) for x in probs.cpu().tolist())
    mean_loss = sum(losses) / max(len(losses), 1)
    return mean_loss, labels, scores


def main() -> int:
    args = parse_args()
    if args.image_size % 14 != 0:
        raise ValueError("DINOv2 ViT-S/14 requires image-size to be a multiple of 14")
    device = resolve_device(args.device)
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {args.data_root} ...")
    records = discover_records(args.data_root)
    records, report = dedupe_records(records, threshold=args.dedupe_threshold)
    write_dedupe_report(args.output_dir / "dedupe_report.json", report)
    print("Kept per split:", report.kept)
    print(f"Removed duplicates: {len(report.removed)} (see dedupe_report.json)")

    train_counts = class_counts(records, "train")
    val_counts = class_counts(records, "val")
    test_counts = class_counts(records, "test")
    print(f"train skip/collect={train_counts}, val={val_counts}, test={test_counts}")

    train_ds = TasteDataset(records, "train", train_transform(args.image_size))
    val_ds = TasteDataset(records, "val", eval_transform(args.image_size))
    train_loader = make_loader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        balanced_sampler=args.balanced_sampler,
        seed=args.seed,
    )
    val_loader = make_loader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(
        backbone_name=args.backbone,
        pretrained=args.pretrained,
        num_classes=2,
        dropout=args.dropout,
        img_size=args.image_size,
    ).to(device)
    set_trainable(model, args.unfreeze_last_blocks, args.full_finetune)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")

    head_params = [p for p in model.head.parameters() if p.requires_grad]
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    groups = [{"params": head_params, "lr": args.lr_head}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": args.lr_backbone})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    weights = class_weights(records, "train").to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"Resumed weights from {args.resume}, starting at epoch {start_epoch}")

    history: list[dict] = []
    best_val_loss = float("inf")
    best_f1 = -1.0
    stale = 0

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", ncols=100)
        for images, targets, _ in progress:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach().cpu()) * images.size(0)
            seen += images.size(0)
            progress.set_postfix(loss=f"{running / max(seen, 1):.4f}")

        scheduler.step()
        val_loss, val_labels, val_scores = evaluate_loss(model, val_loader, criterion, device, amp_enabled)
        val_metrics = binary_metrics(val_labels, val_scores, 0.5)
        train_loss = running / max(seen, 1)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_metrics.items() if k not in {"threshold"}},
        }
        history.append(row)
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f} F1={val_metrics['f1']:.4f}"
        )

        config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        checkpoint = {
            "model": model.state_dict(),
            "class_names": list(CLASS_NAMES),
            "config": config,
            "epoch": epoch,
            "val_metrics": val_metrics,
            "val_loss": val_loss,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_f1 = val_metrics["f1"]
            stale = 0
            torch.save(checkpoint, args.output_dir / "best.pt")
            write_model_config(
                output_path=args.output_dir / "model_config.json",
                checkpoint=checkpoint,
                threshold=0.5,
                version=args.version,
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    save_json(args.output_dir / "history.json", history)
    save_json(
        args.output_dir / "train_summary.json",
        {
            "best_val_loss": best_val_loss,
            "best_f1_at_0.5": best_f1,
            "dedupe": report.__dict__,
            "train_counts": train_counts,
            "val_counts": val_counts,
            "test_counts": test_counts,
            "trainable_params": trainable,
            "total_params": total,
            "amp": amp_enabled,
            "device": str(device),
        },
    )
    print(f"Done. best.pt and model_config.json -> {args.output_dir}")
    print("Next: calibrate_threshold.py on val, then evaluate.py on test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
