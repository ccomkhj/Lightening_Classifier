import torchvision.models as models
from models.base_classifier import BaseClassifier
import torch.nn as nn
import torch

class ResNetClassifier(BaseClassifier):
    resnets = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
        152: models.resnet152,
    }

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
        self.resnet_version = resnet_version
        
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
        
        # Optionally adjust keys if the file was saved without the "resnet_model." prefix.
        sample_key = next(iter(state_dict))
        if not sample_key.startswith("resnet_model."):
            state_dict = {"resnet_model." + key: value for key, value in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()  # Set the model to evaluation mode
        return model