r"""Train DINOv2 ViT-S/14 for skip/suspicious/collect classification on Windows + CUDA.

Configuration:
  training/config.json

Run:
  python training/train.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # tensorboard is optional at runtime
    SummaryWriter = None

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

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
    POSITIVE_INDEX,
    build_model,
    eval_transform,
    set_trainable,
    train_transform,
    write_model_config,
)
from utils import binary_metrics, resolve_device, save_json, seed_everything  # noqa: E402


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (_SCRIPT_DIR / path).resolve()


def load_config() -> SimpleNamespace:
    config_path = _SCRIPT_DIR / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"config not found: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    section = dict(payload.get("train") or {})
    for key in ("data_root", "output_dir"):
        if key in section:
            section[key] = _resolve(section[key])
    output_dir = Path(section.get("output_dir", _SCRIPT_DIR / "runs" / "taste_dinov2_v1"))
    checkpoint_dir = Path(section.get("checkpoint_dir", "ckpt")).expanduser()
    section["checkpoint_dir"] = checkpoint_dir if checkpoint_dir.is_absolute() else output_dir / checkpoint_dir
    tb_dir = section.get("tensorboard_dir")
    if tb_dir:
        section["tensorboard_dir"] = _resolve(tb_dir)
    resume = section.get("resume")
    if resume:
        section["resume"] = _resolve(resume)
    return SimpleNamespace(**section)


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
            probs = torch.softmax(logits.float(), dim=1)[:, POSITIVE_INDEX]
            labels.extend(int(x) for x in targets.cpu().tolist())
            scores.extend(float(x) for x in probs.cpu().tolist())
    mean_loss = sum(losses) / max(len(losses), 1)
    return mean_loss, labels, scores


def main() -> int:
    args = load_config()
    if args.image_size % 14 != 0:
        raise ValueError("DINOv2 ViT-S/14 requires image-size to be a multiple of 14")
    device = resolve_device(args.device)
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {args.data_root} ...")
    records = discover_records(args.data_root, max_image_pixels=args.max_image_pixels)
    records, report = dedupe_records(records, threshold=args.dedupe_threshold)
    write_dedupe_report(args.output_dir / "dedupe_report.json", report)
    print("Kept per split:", report.kept)
    print(f"Removed duplicates: {len(report.removed)} (see dedupe_report.json)")

    train_counts = class_counts(records, "train")
    val_counts = class_counts(records, "val")
    test_counts = class_counts(records, "test")
    print(f"train={train_counts}, val={val_counts}, test={test_counts}")

    if sys.platform == "win32" and args.num_workers > 0:
        print("Windows DataLoader: forcing num_workers=0 to avoid worker crashes.")
        args.num_workers = 0

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
        num_classes=len(CLASS_NAMES),
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
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

    writer = None
    tb_dir = args.tensorboard_dir or args.checkpoint_dir
    if args.tensorboard:
        if SummaryWriter is None:
            print("tensorboard is not installed; continuing without TensorBoard. pip install tensorboard")
        else:
            writer = SummaryWriter(log_dir=str(tb_dir))
            print(f"TensorBoard: {tb_dir}")

    history: list[dict] = []
    best_val_loss = float("inf")
    best_f1 = -1.0
    stale = 0

    for epoch in itertools.count(start_epoch):
        model.train()
        running = 0.0
        seen = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}", ncols=100)
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

        val_loss, val_labels, val_scores = evaluate_loss(model, val_loader, criterion, device, amp_enabled)
        scheduler.step(val_loss)
        val_metrics = binary_metrics(
            [1 if label == POSITIVE_INDEX else 0 for label in val_labels],
            val_scores,
            0.5,
        )
        train_loss = running / max(seen, 1)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_metrics.items() if k not in {"threshold"}},
        }
        history.append(row)
        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            for key in ("precision", "recall", "f1", "auc", "accuracy"):
                if key in val_metrics:
                    writer.add_scalar(f"val/{key}", val_metrics[key], epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
            if len(optimizer.param_groups) > 1:
                writer.add_scalar("lr_backbone", optimizer.param_groups[1]["lr"], epoch)
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
        last_path = args.checkpoint_dir / "last.pt"
        best_path = args.checkpoint_dir / "best.pt"
        torch.save(checkpoint, last_path)
        if args.save_every_epochs > 0 and epoch % args.save_every_epochs == 0:
            torch.save(checkpoint, args.checkpoint_dir / f"epoch_{epoch:04d}.pt")
            print(f"checkpoint: {args.checkpoint_dir / f'epoch_{epoch:04d}.pt'}")
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_f1 = val_metrics["f1"]
            stale = 0
            torch.save(checkpoint, best_path)
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

    if writer is not None:
        writer.add_hparams(
            {
                "lr_head": args.lr_head,
                "lr_backbone": args.lr_backbone,
                "batch_size": args.batch_size,
                "image_size": args.image_size,
                "unfreeze_last_blocks": args.unfreeze_last_blocks,
            },
            {"hparams/best_val_loss": best_val_loss, "hparams/best_f1": best_f1},
        )
        writer.flush()
        writer.close()

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
            "tensorboard_dir": str(tb_dir) if writer is not None else None,
            "checkpoint_dir": str(args.checkpoint_dir),
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
            "save_every_epochs": args.save_every_epochs,
        },
    )
    print(f"Done. checkpoints -> {args.checkpoint_dir}; model_config.json -> {args.output_dir}")
    print("Next: calibrate_threshold.py on val, then evaluate.py on test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
