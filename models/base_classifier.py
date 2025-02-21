import warnings
from pathlib import Path
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
from torchvision import transforms
from torchvision.datasets import ImageFolder
import json
import pandas as pd

warnings.filterwarnings("ignore")

class BaseClassifier(pl.LightningModule):
    optimizers = {"adam": Adam, "sgd": SGD}

    def __init__(
        self,
        num_classes,
        train_path,
        val_path,
        test_path=None,
        optimizer="adam",
        lr=1e-3,
        batch_size=16,
        transfer=True,
        tune_fc_only=True,
        target_size=(730, 968),  # Default size for non-transformer models
    ):
        super().__init__()
        self.num_classes = num_classes
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.lr = lr
        self.batch_size = batch_size
        self.optimizer = self.optimizers[optimizer]
        self.transfer = transfer
        self.tune_fc_only = tune_fc_only
        self.target_size = target_size  # Add target_size attribute
        
        # Loss and metrics
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.precision = Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.recall = Recall(task="multiclass", num_classes=num_classes, average='macro')
        self.f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.test_confusion = ConfusionMatrix(task="multiclass", num_classes=num_classes)

    def _step(self, batch):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        acc = self.acc(preds, y)
        precision = self.precision(preds, y)
        recall = self.recall(preds, y)
        f1 = self.f1(preds, y)
        return loss, acc, precision, recall, f1, preds, y

    def _dataloader(self, data_path, shuffle=False):
        if shuffle:
            transform = transforms.Compose([
                transforms.Resize(self.target_size),  # Use target_size
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(self.target_size),  # Use target_size
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
            ])
        
        dataset = ImageFolder(root=data_path, transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=4)

    def training_step(self, batch, batch_idx):
        loss, acc, precision, recall, f1, _, _ = self._step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_precision", precision, on_step=True, on_epoch=True)
        self.log("train_recall", recall, on_step=True, on_epoch=True)
        self.log("train_f1", f1, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, precision, recall, f1, _, _ = self._step(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        self.log("val_precision", precision, on_epoch=True)
        self.log("val_recall", recall, on_epoch=True)
        self.log("val_f1", f1, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss, acc, precision, recall, f1, preds, y = self._step(batch)
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", acc, on_epoch=True)
        self.log("test_precision", precision, on_epoch=True)
        self.log("test_recall", recall, on_epoch=True)
        self.log("test_f1", f1, on_epoch=True)
        self.test_confusion.update(preds, y)
    
    def on_test_epoch_end(self):
        # Compute confusion matrix
        cm = self.test_confusion.compute()
        
        # Get class names from the dataset
        dataset = ImageFolder(root=self.test_path)
        class_names = dataset.classes
        
        # Save confusion matrix as CSV with class names
        cm_np = cm.cpu().numpy()
        df = pd.DataFrame(cm_np,
                          columns=class_names,
                          index=class_names)
        
        # Save under the corresponding version directory
        version_dir = Path(self.trainer.log_dir)  # Automatically points to the correct version directory
        version_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        csv_path = version_dir / "test_confusion_matrix.csv"
        df.to_csv(csv_path)
        
        # Save model configuration as JSON
        config = {
            "classifier": self.__class__.__name__,
            "num_classes": self.num_classes,
            "train_path": self.train_path.__str__(),
            "val_path": self.val_path.__str__(),
            "test_path": self.test_path.__str__(),
            "optimizer": self.optimizer.__name__,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "transfer": self.transfer,
            "tune_fc_only": self.tune_fc_only,
            "class_names": class_names,  # Include class names in the config
        }
        config_path = version_dir / "model_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        
        # Reset confusion matrix
        self.test_confusion.reset()
        
    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def train_dataloader(self):
        return self._dataloader(self.train_path, shuffle=True)

    def val_dataloader(self):
        return self._dataloader(self.val_path)

    def test_dataloader(self):
        return self._dataloader(self.test_path)