import warnings
from pathlib import Path
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Precision, Recall, F1Score
from torchvision import transforms
from torchvision.datasets import ImageFolder
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

warnings.filterwarnings("ignore")

class ResNetClassifier(pl.LightningModule):
    resnets = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
        152: models.resnet152,
    }
    optimizers = {"adam": Adam, "sgd": SGD}

    def __init__(
        self,
        num_classes,
        resnet_version,
        train_path,
        val_path,
        test_path=None,
        optimizer="adam",
        lr=1e-3,
        batch_size=16,
        transfer=True,
        tune_fc_only=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.lr = lr
        self.batch_size = batch_size
        self.optimizer = self.optimizers[optimizer]
        
        # Loss and metrics
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.precision = Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.recall = Recall(task="multiclass", num_classes=num_classes, average='macro')
        self.f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')
        
        # ResNet model setup
        self.resnet_model = self.resnets[resnet_version](pretrained=transfer)
        linear_size = self.resnet_model.fc.in_features
        self.resnet_model.fc = nn.Linear(linear_size, num_classes)
        
        # Freeze layers if needed
        if tune_fc_only:
            for param in self.resnet_model.parameters():
                param.requires_grad = False
            for param in self.resnet_model.fc.parameters():
                param.requires_grad = True

    def forward(self, X):
        return self.resnet_model(X)

    def configure_optimizers(self):
        # Create the optimizer instance.
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        # Define a StepLR scheduler that decays the learning rate every 10 epochs by a factor of 0.1.
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        # Return the optimizer and scheduler dictionary.
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # or 'step', if you want to update every step.
                "frequency": 1,
            },
        }

    def _step(self, batch):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        acc = self.acc(preds, y)
        precision = self.precision(preds, y)
        recall = self.recall(preds, y)
        f1 = self.f1(preds, y)
        return loss, acc, precision, recall, f1

    def _dataloader(self, data_path, shuffle=False):
        resize_size = (1460, 1936)
        
        if shuffle:
            # For training data, introduce dynamic augmentation using RandAugment.
            transform = transforms.Compose([
                transforms.Resize(resize_size),
                transforms.RandAugment(num_ops=2, magnitude=9),  # Adjust num_ops and magnitude as needed.
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            # For validation and testing, use deterministic transforms.
            transform = transforms.Compose([
                transforms.Resize(resize_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])
        
        dataset = ImageFolder(root=data_path, transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=4)

    # Training methods
    def training_step(self, batch, batch_idx):
        loss, acc, precision, recall, f1 = self._step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_precision", precision, on_step=True, on_epoch=True)
        self.log("train_recall", recall, on_step=True, on_epoch=True)
        self.log("train_f1", f1, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, precision, recall, f1 = self._step(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        self.log("val_precision", precision, on_epoch=True)
        self.log("val_recall", recall, on_epoch=True)
        self.log("val_f1", f1, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss, acc, precision, recall, f1 = self._step(batch)
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", acc, on_epoch=True)
        self.log("test_precision", precision, on_epoch=True)
        self.log("test_recall", recall, on_epoch=True)
        self.log("test_f1", f1, on_epoch=True)
    
    # Data loaders
    def train_dataloader(self):
        return self._dataloader(self.train_path, shuffle=True)

    def val_dataloader(self):
        return self._dataloader(self.val_path)

    def test_dataloader(self):
        return self._dataloader(self.test_path)