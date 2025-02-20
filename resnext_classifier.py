import torchvision.models as models
import torch.nn as nn
from base_classifier import BaseClassifier

class ResNextClassifier(BaseClassifier):
    resnexts = {
        50: models.resnext50_32x4d,
        101: models.resnext101_32x8d,
    }

    def __init__(
        self,
        num_classes,
        resnext_version,
        train_path,
        val_path,
        test_path=None,
        optimizer="adam",
        lr=1e-3,
        batch_size=16,
        transfer=True,
        tune_fc_only=True,
    ):
        super().__init__(
            num_classes=num_classes,
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            optimizer=optimizer,
            lr=lr,
            batch_size=batch_size,
            transfer=transfer,
            tune_fc_only=tune_fc_only,
        )
        self.resnext_version = resnext_version
        
        # ResNext model setup
        self.resnext_model = self.resnexts[resnext_version](pretrained=transfer)
        linear_size = self.resnext_model.fc.in_features
        self.resnext_model.fc = nn.Linear(linear_size, num_classes)
        
        # Freeze layers if needed
        if tune_fc_only:
            for param in self.resnext_model.parameters():
                param.requires_grad = False
            for param in self.resnext_model.fc.parameters():
                param.requires_grad = True

    def forward(self, X):
        return self.resnext_model(X)