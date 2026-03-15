from pathlib import Path
import os
import warnings

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torchvision
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from torch.utils.data import DataLoader

# Import thêm các metrics cần thiết cho Lane Segmentation
from torchmetrics.classification import JaccardIndex, Accuracy, Precision, Recall, F1Score

# seed everything lightning
pl.seed_everything(42, workers=True)
from dataset import ORIG_H, ORIG_W
from visualize import plot_metrics

def create_model(freeze_backbone=False, num_classes=3):
    weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights, aux_loss=True)
    model.classifier[-1] = nn.Conv2d(256, num_classes, 1)
    model.aux_classifier[-1] = nn.Conv2d(256, num_classes, 1)
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
    return model

def _safe_float(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.numel() == 1:
            return float(x.item())
        return [float(v) for v in x.flatten().tolist()]
    if isinstance(x, (np.floating, float, int)):
        return float(x)
    return x

def _format_metric_line(name, value, width=18):
    if isinstance(value, float):
        return f"{name:<{width}}: {value:.4f}"
    return f"{name:<{width}}: {value}"

def _print_benchmark_block(title, metrics):
    print(f"\n{title}")
    print("-" * len(title))
    for k, v in metrics.items():
        print(_format_metric_line(k, v))

class MetricsHistoryCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.train_loss = []
        self.val_loss = []
        self.val_miou = []
        # Thêm mảng lưu lịch sử các metrics mới
        self.val_precision = []
        self.val_recall = []
        self.val_f1 = []

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        self.train_loss.append(_safe_float(metrics.get("train_loss", np.nan)))
        self.val_loss.append(_safe_float(metrics.get("val_loss", np.nan)))
        self.val_miou.append(_safe_float(metrics.get("val_miou", np.nan)))
        
        self.val_precision.append(_safe_float(metrics.get("val_precision", np.nan)))
        self.val_recall.append(_safe_float(metrics.get("val_recall", np.nan)))
        self.val_f1.append(_safe_float(metrics.get("val_f1", np.nan)))

class LaneSegLightningModule(pl.LightningModule):
    def __init__(
        self,
        model,
        num_classes=3,
        lr=1e-4,
        weight_decay=1e-3,
        betas=(0.9, 0.98),
        aux_weight=0.4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        
        # --- KHAI BÁO METRICS CHO VALIDATION ---
        self.val_iou = JaccardIndex(task="multiclass", num_classes=num_classes, average=None, ignore_index=-1)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes, average="macro", ignore_index=-1)
        self.val_precision = Precision(task="multiclass", num_classes=num_classes, average="macro", ignore_index=-1)
        self.val_recall = Recall(task="multiclass", num_classes=num_classes, average="macro", ignore_index=-1)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro", ignore_index=-1)

        # --- KHAI BÁO METRICS CHO TEST ---
        self.test_iou = JaccardIndex(task="multiclass", num_classes=num_classes, average=None, ignore_index=-1)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes, average="macro", ignore_index=-1)
        self.test_precision = Precision(task="multiclass", num_classes=num_classes, average="macro", ignore_index=-1)
        self.test_recall = Recall(task="multiclass", num_classes=num_classes, average="macro", ignore_index=-1)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro", ignore_index=-1)

    def forward(self, x):
        return self.model(x)

    def _loss_and_logits(self, x, y):
        outputs = self.model(x)
        main_out = outputs["out"]
        loss = self.loss_fn(main_out, y)
        if "aux" in outputs:
            loss = loss + self.hparams.aux_weight * self.loss_fn(outputs["aux"], y)
        return loss, main_out

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, _ = self._loss_and_logits(x, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, out = self._loss_and_logits(x, y)
        pred = torch.argmax(out, dim=1)

        pred_crop = pred[:, :ORIG_H, :ORIG_W]
        out_crop = out[:, :, :ORIG_H, :ORIG_W]
        y_crop = y[:, :ORIG_H, :ORIG_W]

        # Update tất cả các metrics
        self.val_iou.update(pred_crop, y_crop)
        self.val_acc.update(pred_crop, y_crop)
        self.val_precision.update(pred_crop, y_crop)
        self.val_recall.update(pred_crop, y_crop)
        self.val_f1.update(pred_crop, y_crop)

        val_loss = self.loss_fn(out_crop, y_crop)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0))

    def on_validation_epoch_end(self):
        # Reset nếu đang trong bước sanity check của Lightning
        if self.trainer.sanity_checking:
            self.val_iou.reset()
            self.val_acc.reset()
            self.val_precision.reset()
            self.val_recall.reset()
            self.val_f1.reset()
            return
            
        iou_t = self.val_iou.compute()
        iou_t = torch.where(iou_t < 0, torch.zeros_like(iou_t), iou_t)
        miou = iou_t.mean()

        # Log tổng hợp
        self.log("val_miou", miou, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc.compute(), on_step=False, on_epoch=True)
        self.log("val_precision", self.val_precision.compute(), on_step=False, on_epoch=True)
        self.log("val_recall", self.val_recall.compute(), on_step=False, on_epoch=True)
        self.log("val_f1", self.val_f1.compute(), on_step=False, on_epoch=True, prog_bar=True) # Hiện F1 lên thanh progress bar

        for i, cls_iou in enumerate(iou_t):
            self.log(f"val_iou_class_{i}", cls_iou, on_step=False, on_epoch=True)
            
        # Bắt buộc reset sau khi epoch kết thúc
        self.val_iou.reset()
        self.val_acc.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        _, out = self._loss_and_logits(x, y)
        pred = torch.argmax(out, dim=1)

        pred_crop = pred[:, :ORIG_H, :ORIG_W]
        out_crop = out[:, :, :ORIG_H, :ORIG_W]
        y_crop = y[:, :ORIG_H, :ORIG_W]

        self.test_iou.update(pred_crop, y_crop)
        self.test_acc.update(pred_crop, y_crop)
        self.test_precision.update(pred_crop, y_crop)
        self.test_recall.update(pred_crop, y_crop)
        self.test_f1.update(pred_crop, y_crop)
        
        test_loss = self.loss_fn(out_crop, y_crop)
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0))

    def on_test_epoch_end(self):
        iou_t = self.test_iou.compute()
        iou_t = torch.where(iou_t < 0, torch.zeros_like(iou_t), iou_t)
        miou = iou_t.mean()

        self.log("test_miou", miou, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_acc.compute(), on_step=False, on_epoch=True)
        self.log("test_precision", self.test_precision.compute(), on_step=False, on_epoch=True)
        self.log("test_recall", self.test_recall.compute(), on_step=False, on_epoch=True)
        self.log("test_f1", self.test_f1.compute(), on_step=False, on_epoch=True)

        for i, cls_iou in enumerate(iou_t):
            self.log(f"test_iou_class_{i}", cls_iou, on_step=False, on_epoch=True)
            
        self.test_iou.reset()
        self.test_acc.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            x, y = batch
        else:
            x, y = batch, None

        out = self.model(x)["out"]
        pred = torch.argmax(out, dim=1)

        pred_crop = pred[:, :ORIG_H, :ORIG_W]
        result = {"pred": pred_crop.detach().cpu()}

        if y is not None:
            y_crop = y[:, :ORIG_H, :ORIG_W].detach().cpu()
            result["gt"] = y_crop

        return result

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=self.hparams.betas,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=10, min_lr=1e-6
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_miou",
                "interval": "epoch",
                "frequency": 1,
            },
        }

class LaneSegDataModule(pl.LightningDataModule):
    def __init__(self, train_ds, val_ds, test_ds=None, batch_size=16, num_workers=2, pin_memory=True, persistent_workers=True):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Sửa logic chỗ này một chút để an toàn tuyệt đối với DataLoader
        self.persistent_workers = persistent_workers if self.num_workers > 0 else False
        
        if os.name == "nt" and self.num_workers > 0:
            warnings.warn(
                "Windows + large in-memory dataset can fail with DataLoader multiprocessing. "
                "Falling back to num_workers=0.",
                RuntimeWarning,
            )
            self.num_workers = 0
            self.persistent_workers = False
        self.pin_memory = pin_memory

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        if self.test_ds is None:
            return None
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    @property
    def has_test(self):
        return self.test_ds is not None

def _resolve_precision(accelerator):
    return "bf16-mixed" if accelerator == "gpu" else "32-true"

def benchmark_model(model, dataloader, device, split="test", num_classes=3):
    lit_model = LaneSegLightningModule(model=model, num_classes=num_classes)
    trainer = pl.Trainer(
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )

    split = split.lower()
    if split == "val":
        results = trainer.validate(lit_model, dataloaders=dataloader, verbose=False)
    else:
        results = trainer.test(lit_model, dataloaders=dataloader, verbose=False)

    metrics = {k: _safe_float(v) for k, v in (results[0] if results else {}).items()}
    print(f"{split.upper()} benchmark:", metrics)
    return metrics


def train_model(model, datamodule, EPOCHS, FILE_NAME, num_classes=3):
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    history_callback = MetricsHistoryCallback()

    checkpoint_dir = Path("model_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_path = checkpoint_dir / Path(FILE_NAME).name

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=f"{output_path.stem}" + "-lightning-{epoch:02d}-{val_miou:.4f}",
        monitor="val_miou",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    early_stopping = EarlyStopping(
        monitor="val_miou",
        mode="max",
        patience=20,
        min_delta=5e-4,
    )

    bar_cb = TQDMProgressBar(leave=True)
    lit_model = LaneSegLightningModule(model=model, num_classes=num_classes)
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator=accelerator,
        devices=1,
        precision=_resolve_precision(accelerator),
        callbacks=[history_callback, checkpoint_callback, early_stopping, bar_cb, LearningRateMonitor(logging_interval="epoch")],
        log_every_n_steps=10,
    )

    trainer.fit(lit_model, datamodule=datamodule)

    best_ckpt = checkpoint_callback.best_model_path
    best_score = checkpoint_callback.best_model_score

    best_model = model
    best_lit_model = lit_model
    if best_ckpt:
        best_lit_model = LaneSegLightningModule.load_from_checkpoint(
            best_ckpt,
            model=create_model(num_classes=num_classes),
            num_classes=num_classes,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        best_model = best_lit_model.model.to(device).eval()
        torch.save(best_model.state_dict(), str(output_path))

    best_miou = _safe_float(best_score) if best_score is not None else float("nan")
    print("\nTraining completed")
    print(_format_metric_line("best_miou", best_miou))
    print(_format_metric_line("weights_file", str(output_path)))

    plot_metrics(history_callback.train_loss, history_callback.val_loss, history_callback.val_miou)

    results = {
        "best_miou": best_miou,
        "best_checkpoint": best_ckpt,
        "weights_file": str(output_path),
    }

    eval_trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        precision=_resolve_precision(accelerator),
    )

    val_results = eval_trainer.validate(model=best_lit_model, datamodule=datamodule, verbose=False)
    results["val_benchmark"] = {k: _safe_float(v) for k, v in (val_results[0] if val_results else {}).items()}
    _print_benchmark_block("Validation benchmark", results["val_benchmark"])

    if getattr(datamodule, "has_test", False):
        test_results = eval_trainer.test(model=best_lit_model, datamodule=datamodule, verbose=False)
        results["test_benchmark"] = {k: _safe_float(v) for k, v in (test_results[0] if test_results else {}).items()}
        _print_benchmark_block("Test benchmark", results["test_benchmark"])

    return results