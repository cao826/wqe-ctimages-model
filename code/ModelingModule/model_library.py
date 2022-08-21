"""
Module Level Docstring
"""
import torch
from  torch import nn
from torch.nn import functional as F


def forward_through(linear_layer, activation, input_tensor):
    """Applies the layer and the activation to the input tensor"""
    return activation(linear_layer(input_tensor))

class NlstModel(nn.Module):
    """First version of model that I designed.

    I have to try this with other backbones.
    At the very least, a model with less parameters.
    This model is overfitting like crazy.
    """
    def __init__(self):
        """
        """
        super().__init__(use_leaky_relu=False)
        self.backbone = torch.hub.load(
            repo_or_dir='pytorch/vision:v0.10.0',
            model='resnet101',
            pretrained=True
        )
        self.get_number_backbone_params()

        if use_leaky_relu:
            self.change_to_leaky_relu()

        self.linear_1 = nn.Linear(in_features=1004, #this is a magic number
                                  out_features=100  #these are all magic numbers :(
)
        self.linear_2 = nn.Linear(in_features=100,
                                  out_features=10)

        self.linear_3 = nn.Linear(in_features=10,
                                  out_features=2)

        self.dropout = nn.Dropout(p=.5)

    def change_to_leaky_relu(self):
        """Changes the activation functions to LReLU"""
        ### change ReLU to leakyReLU ###
        for _, module in self.backbone.named_modules():
            if hasattr(module, 'relu'):
                module.relu = nn.LeakyReLU(inplace=True)

    def get_number_backbone_params(self):
        """Returns the number of params the backbone has"""
        count = 0
        for _ in self.backbone.parameters():
            count += 1
        self.backbone_params_number = count

    def freeze_all_params_except(self, last_n_layers=10, freeze_all=False):
        """Freezes all params except the last N layers"""
        if freeze_all:
            for param in self.backbone.parameters():
                param.requires_grad = False
        too_large_error = 'More params to unfreeze than number of parameters'
        assert last_n_layers < self.backbone_params_number, too_large_error
        unfreeze_threshold = self.backbone_params_number - last_n_layers
        for i, param in enumerate(self.backbone.parameters()):
            if i >= unfreeze_threshold:
                continue
            param.requires_grad = False

    def forward(self, x, clinical_info):
        """Function where the model predicts classes"""
        image_feature_map = self.backbone(x)
        combined_feature_tensor = torch.cat((clinical_info, image_feature_map),
                                            dim=1)

        x = self.dropout(F.leaky_relu(self.linear_1(combined_feature_tensor)))
        x = self.dropout(F.leaky_relu(self.linear_2(x)))
        x = self.linear_3(x)

        return x
