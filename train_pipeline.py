from __future__ import annotations

import csv
import json
import random
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import BDDDataset, MEAN, ORIG_H, ORIG_W, STD, load_npy_pair, split_data
from trainer import LaneSegDataModule, LaneSegLightningModule, create_model, train_model
from visualize import COLOR_MAP


FILE_NAME = "checkpoints/best.pt"
SEED = 42
NUM_CLASSES = 3
EPOCHS = 100


def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_output_dirs(base_dir: str = "outputs") -> dict[str, Path]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"final_driverlane_seg_{ts}"

    paths = {
        "run": run_dir,
        "images": run_dir / "images",
        "predictions": run_dir / "predictions",
        "metrics": run_dir / "metrics",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def denormalize_image(img_chw: torch.Tensor) -> np.ndarray:
    img = img_chw * STD + MEAN
    img = img.clamp(0, 1)
    img = (img * 255).permute(1, 2, 0).byte().cpu().numpy()
    return img[:ORIG_H, :ORIG_W]


def save_dataset_preview(images: np.ndarray, labels: np.ndarray, out_dir: Path, k: int = 5) -> None:
    k = min(k, len(images))
    for i in range(k):
        image = np.ascontiguousarray(images[i])
        mask = labels[i].copy()
        mask[mask < 0] = 2
        mask = mask.astype(np.uint8)

        mask_rgb = COLOR_MAP[mask]
        overlay_rgb = cv2.addWeighted(image, 1.0, mask_rgb, 0.5, 0.0)

        fig = plt.figure(figsize=(12, 4))
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.set_title("Image")
        ax1.imshow(image)
        ax1.axis("off")

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.set_title("Mask")
        ax2.imshow(mask, cmap="gray", vmin=0, vmax=2)
        ax2.axis("off")

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.set_title("Overlay")
        ax3.imshow(overlay_rgb)
        ax3.axis("off")

        fig.tight_layout()
        fig.savefig(out_dir / f"train_preview_{i:03d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def copy_latest_lightning_metrics(metrics_dir: Path) -> Path | None:
    logs_root = Path("lightning_logs")
    if not logs_root.exists():
        return None

    csv_files = sorted(logs_root.glob("version_*/metrics.csv"), key=lambda p: p.stat().st_mtime)
    if not csv_files:
        return None

    latest_csv = csv_files[-1]
    target = metrics_dir / "lightning_metrics.csv"
    shutil.copy2(latest_csv, target)
    return target


def save_metrics_summary_from_csv(csv_path: Path, metrics_dir: Path) -> None:
    rows: list[dict[str, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        return

    best_val_miou = float("-inf")
    best_row: dict[str, str] | None = None
    for r in rows:
        raw = r.get("val_miou")
        if not raw:
            continue
        try:
            v = float(raw)
        except ValueError:
            continue
        if v > best_val_miou:
            best_val_miou = v
            best_row = r

    summary = {
        "rows": len(rows),
        "best_val_miou": None if best_row is None else best_val_miou,
        "best_row": best_row,
        "latest_row": rows[-1],
    }

    (metrics_dir / "lightning_metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def save_train_results(results: dict, metrics_dir: Path) -> None:
    (metrics_dir / "train_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def save_prediction_visualizations(
    file_name: str,
    val_ds,
    datamodule: LaneSegDataModule,
    out_dir: Path,
    num_classes: int = NUM_CLASSES,
    k: int = 5,
) -> None:
    model = create_model(num_classes=num_classes)
    lit_model = LaneSegLightningModule(model=model, num_classes=num_classes)
    lit_model.model.load_state_dict(torch.load(file_name, map_location="cpu"))

    predict_trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        deterministic="warn",
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
    )

    pred_batches = predict_trainer.predict(
        model=lit_model,
        dataloaders=datamodule.val_dataloader(),
    )
    all_pred = torch.cat([b["pred"] for b in pred_batches], dim=0)

    n = min(k, all_pred.shape[0])
    for i in range(n):
        img, gt_t = val_ds[i]
        pred = all_pred[i].numpy()

        pred = pred[:ORIG_H, :ORIG_W]
        gt = gt_t[:ORIG_H, :ORIG_W].cpu().numpy()
        gt[gt < 0] = 2

        pred_rgb = COLOR_MAP[pred]
        gt_rgb = COLOR_MAP[gt]

        img_denorm = denormalize_image(img)
        overlay_pred = cv2.addWeighted(img_denorm, 1.0, pred_rgb, 0.5, 0)
        overlay_gt = cv2.addWeighted(img_denorm, 1.0, gt_rgb, 0.5, 0)

        cv2.imwrite(str(out_dir / f"sample_{i:03d}_image.png"), cv2.cvtColor(img_denorm, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"sample_{i:03d}_pred_mask.png"), cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"sample_{i:03d}_gt_mask.png"), cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"sample_{i:03d}_overlay_pred.png"), cv2.cvtColor(overlay_pred, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"sample_{i:03d}_overlay_gt.png"), cv2.cvtColor(overlay_gt, cv2.COLOR_RGB2BGR))

        fig = plt.figure(figsize=(20, 5))
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.set_title("Image")
        ax1.imshow(img_denorm)
        ax1.axis("off")

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.set_title("Overlay Ground Truth")
        ax2.imshow(overlay_gt)
        ax2.axis("off")

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.set_title("Overlay Prediction")
        ax3.imshow(overlay_pred)
        ax3.axis("off")

        fig.tight_layout()
        fig.savefig(out_dir / f"sample_{i:03d}_compare.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    set_seed(SEED)
    torch.set_float32_matmul_precision("medium")

    output_dirs = prepare_output_dirs(base_dir="outputs")

    images, labels = load_npy_pair(
        "data/train/image_180_320.npy",
        "data/train/label_180_320.npy",
        mmap_mode="r",
    )

    test_images, test_labels = load_npy_pair(
        "data/val/image_180_320.npy",
        "data/val/label_180_320.npy",
        mmap_mode="r",
    )

    test_ds = BDDDataset(test_images, test_labels)
    full_ds = BDDDataset(images, labels)
    train_ds, val_ds = split_data(full_ds, train_size=60000, seed=SEED)

    print(
        "Data size:",
        len(full_ds) + len(test_ds),
        "| Train:",
        len(train_ds),
        "| Val:",
        len(val_ds),
        "| Test:",
        len(test_ds),
    )

    save_dataset_preview(images, labels, output_dirs["images"], k=5)

    datamodule = LaneSegDataModule(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        batch_size=128,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=NUM_CLASSES).to(device)

    results = train_model(
        model=model,
        datamodule=datamodule,
        EPOCHS=EPOCHS,
        FILE_NAME=FILE_NAME,
        num_classes=NUM_CLASSES,
    )

    save_train_results(results, output_dirs["metrics"])

    latest_csv = copy_latest_lightning_metrics(output_dirs["metrics"])
    if latest_csv is not None:
        save_metrics_summary_from_csv(latest_csv, output_dirs["metrics"])

    save_prediction_visualizations(
        file_name=FILE_NAME,
        val_ds=val_ds,
        datamodule=datamodule,
        out_dir=output_dirs["predictions"],
        num_classes=NUM_CLASSES,
        k=5,
    )

    print(f"Saved outputs to: {output_dirs['run']}")


if __name__ == "__main__":
    main()
