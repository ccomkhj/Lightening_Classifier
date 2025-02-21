import warnings
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
from torchvision import transforms
from torchvision.datasets import ImageFolder
import pandas as pd
from pathlib import Path
import json
import os

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
        # Metrics will accumulate state for the whole epoch.
        self.precision = Precision(task="multiclass", num_classes=num_classes, average='weighted')
        self.recall = Recall(task="multiclass", num_classes=num_classes, average='weighted')
        self.f1 = F1Score(task="multiclass", num_classes=num_classes, average='weighted')
        self.test_confusion = ConfusionMatrix(task="multiclass", num_classes=num_classes)

    def _step(self, batch):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        # Update metrics (stateful accumulation)
        self.acc.update(preds, y)
        self.precision.update(preds, y)
        self.recall.update(preds, y)
        self.f1.update(preds, y)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        # Log the learning rate each step (assumes one optimizer)
        opt = self.trainer.optimizers[0]
        lr = opt.param_groups[0]['lr']
        self.log("lr", lr, on_step=True, on_epoch=True, prog_bar=True)

        loss, preds, y = self._step(batch)
        # We can log batch-level training loss; if aggregated training metrics are desired, 
        # you may create separate metric objects and log them in training_epoch_end.
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self._step(batch)
        # Log batch loss so that the average over the epoch is computed automatically
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        # Note: We're not logging precision, recall, or F1 here because we want the aggregated
        # epoch-level values computed in on_validation_epoch_end.
        return loss

    def on_validation_epoch_end(self):
        # Compute aggregated metrics for the entire validation epoch.
        epoch_precision = self.precision.compute()
        epoch_recall = self.recall.compute()
        epoch_f1 = self.f1.compute()
        epoch_acc = self.acc.compute()
        
        self.log("val_precision", epoch_precision, prog_bar=True)
        self.log("val_recall", epoch_recall, prog_bar=True)
        self.log("val_f1", epoch_f1, prog_bar=True)
        self.log("val_acc", epoch_acc, prog_bar=True)
        
        # Reset metrics at the end of the epoch.
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        self.acc.reset()

    def test_step(self, batch, batch_idx):
        loss, preds, y = self._step(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        # Update the confusion metric separately.
        self.test_confusion.update(preds, y)
        return loss
    
    def _save_confusion_matrix(self, cm_np, class_names):
        # Create new labels for rows and columns.
        actual_labels = [f"Actual_{cls}" for cls in class_names]
        predicted_labels = [f"Predicted_{cls}" for cls in class_names]
        
        # Create a DataFrame using the modified labels.
        df = pd.DataFrame(cm_np, columns=predicted_labels, index=actual_labels)
        
        # Optionally, you can set a name for the index.
        df.index.name = "Actual Classes"
        
        # Get the log directory for the current trainer version.
        version_dir = Path(self.trainer.log_dir)  # Typically something like lightning_logs/version_x
        version_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        
        # Define the path for saving the confusion matrix CSV.
        csv_path = version_dir / "test_confusion_matrix.csv"
        
        # Save the DataFrame as a CSV file.
        df.to_csv(csv_path)
            
    def on_test_epoch_end(self):
        # Compute aggregated test metrics using torchmetrics objects (excluding confusion matrix)
        test_precision = self.precision.compute()
        test_recall = self.recall.compute()
        test_f1 = self.f1.compute()
        test_acc = self.acc.compute()
        
        self.log("test_precision", test_precision)
        self.log("test_recall", test_recall)
        self.log("test_f1", test_f1)
        self.log("test_acc", test_acc)
        
        # Get class names from the dataset
        dataset = ImageFolder(root=self.test_path)
        class_names = dataset.classes
        
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
        config_path = os.path.join(self.logger.log_dir, "model_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        
        # Compute confusion matrix-based metrics, if it becomes same with above, we can delete below.
        cm = self.test_confusion.compute()
        cm_np = cm.cpu().numpy()

        # Reset test metrics
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        self.acc.reset()
        # Save confusion matrix and reset
        self._save_confusion_matrix(cm_np, class_names)
        self.test_confusion.reset()

    def _dataloader(self, data_path, shuffle=False):
        if shuffle:
            transform = transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        dataset = ImageFolder(root=data_path, transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=4)

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