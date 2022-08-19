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
        self.get_number_backbone_params()
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
    def get_number_backbone_params(self):
        """Returns the number of params the backbone has"""
        count = 0
        for param in self.backbone.parameters():
            count += 1
        self.backbone_params_number = count

    def freeze_all_params_except(self, last_n_layers=10, freeze_all=False):
        """Freezes all params except the last N layers"""
        if freeze_all:
            for param in self.backbone.parameters():
                param.requires_gradd = False
        too_large_error = 'More params to unfreeze than number of parameters'
        assert last_n_layers < self.backbone_params_number, too_large_error
        unfreeze_threshold = self.backbone_params_number - last_n_layers
        for i, param in enumerate(self.backbone.parameters()):
            if i >= unfreeze_threshold:
                continue
            else:
                param.requires_grad = False

    def forward(self, x, clinical_info):
        """
        """
        image_feature_map = self.backbone(x)
        combined_feature_tensor = torch.cat((clinical_info, image_feature_map), dim=1)

        x = self.dropout(F.leaky_relu(self.fc1(combined_feature_tensor)))
        x = self.dropout(F.leaky_relu(self.fc2(x)))
        x = self.fc_out(x)

        return x
