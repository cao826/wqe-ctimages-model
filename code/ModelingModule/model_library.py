import torch
import torch.nn as nn
import torch.nn.functional as F

class NlstModel(nn.Module):
    """
    """
    def __init__(self):
        """
        """
        super().__init__()
        self.backbone = torch.hub.load(
            repo_or_dir='pytorch/vision:v0.10.0',
            model='resnet101',
            pretrained=True
        )
        ### change ReLU to leakyReLU ###
        for name, module in self.backbone.named_modules():
            if hasattr(module, 'relu'):
                module.relu = nn.LeakyReLU(inplace=True)
        self.fc1 = nn.Linear(
            in_features=1004,
            out_features=100
        )
        self.fc2 = nn.Linear(
            in_features=100,
            out_features=10
        )
        self.fc_out = nn.Linear(
            in_features=10,
            out_features=2
        )
        self.dropout = nn.Dropout(p=.5)
    
    def forward(self, x, clinical_info):
        """
        """
        image_feature_map = self.backbone(x)
        combined_feature_tensor = torch.cat((clinical_info, image_feature_map), dim=1)

        x = self.dropout(F.leaky_relu(self.fc1(combined_feature_tensor)))
        x = self.dropout(F.leaky_relu(self.fc2(x)))
        x = self.fc_out(x)

        return x
