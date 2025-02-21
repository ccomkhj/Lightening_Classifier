import torchvision.models as models
import torch.nn as nn
import torch
from models.base_classifier import BaseClassifier

class ResNextClassifier(BaseClassifier):
    resnexts = {
        50: models.resnext50_32x4d,
        101: models.resnext101_64x4d,
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
        target_size=(730, 968),
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
            target_size=target_size
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
    
    @classmethod
    def load_model(cls, model_weight_path, **kwargs):
        """
        Creates an instance of the model and loads the weights from a checkpoint.
        
        Args:
          model_weight_path (str): The file path to the saved weights.
          **kwargs: All other keyword args required to instantiate the model (e.g., num_classes,
                    train_path, etc.).
                    
        Returns:
          An instance of Resnet in evaluation mode.
        """
        # Instantiate the model with provided kwargs
        model = cls(**kwargs)
        
        # Load the saved state dictionary
        state_dict = torch.load(model_weight_path, map_location="cpu")
        
        # Optionally adjust keys if the file was saved without the "resnext_model." prefix.
        sample_key = next(iter(state_dict))
        if not sample_key.startswith("resnext_model."):
            state_dict = {"resnext_model." + key: value for key, value in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()  # Set the model to evaluation mode
        return model