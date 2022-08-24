"""
Module Level Docstring
"""
from collections import namedtuple
from typing import List, Dict, NamedTuple
import torch
from  torch import nn
from torch.nn import functional as F

ModelInfo = namedtuple("ModelInfo", "name repo")

torchvision_model_repo = "pytorch/vision:v0.10.1"

known_models = {
    'resnet101': ModelInfo(
        name='resnet101',
        repo=torchvision_model_repo
        ),
    'resnet50': ModelInfo(
        name='',
        repo=''
        ),
    'mobilenet': ModelInfo(
        name='lraspp_mobilenet_v3_large',
        repo=torchvision_model_repo
        ),
    'resnext': ModelInfo(
        name='resnext50_32x4d',
        repo=torchvision_model_repo),
    'efficientnet': ModelInfo(
        name='nvidia_efficientnet_b0',
        repo='NVIDIA/DeepLearningExamples:torchhub'),
    'inception': ModelInfo(
        name='inception_v3',
        repo=torchvision_model_repo
        )
}

def forward_through(linear_layer, activation, input_tensor, dropout=None):
    """Applies the layer and the activation to the input tensor"""
    output = None

    if dropout and not activation:
        raise Exception('Invalid dropout and activation input')

    if dropout:
        output = dropout(activation(linear_layer(input_tensor)))

    if activation:
        output = activation(linear_layer(input_tensor))

    if not dropout and not activation:
        output = linear_layer(input_tensor)

    return output

def create_linear_layer(in_features, out_features):
    """Creates nn.Linear layer"""
    return nn.Linear(in_features=in_features,
                     out_features=out_features)

def create_network(feature_counts):
    """Creates linear layers form first to last feature count"""
    layers = []
    #print(len(feature_counts))
    for i in range( len(feature_counts) - 1):
        #print(i)
        in_features = feature_counts[i]
        out_features = feature_counts[i+1]
        layers.append(create_linear_layer(in_features=in_features,
                                          out_features=out_features))
    return layers

def load_model(model_info, pretrained: bool):
    """Loads a model in. Can be pretrained or not"""
    return torch.hub.load(repo_or_dir=model_info.repo,
                          model=model_info.name,
                          pretrained=pretrained)

class BackboneGetter():
    """Class level docstring"""
    def __init__(self, model_dict: Dict[str,NamedTuple]):
        """Constructor"""
        self.model_dict = model_dict
    def list_models(self):
        """prints all the models available"""
        for key in self.model_dict:
            print(key)
    def __call__(self, model_name, pretrained=True):
        """Loads specified model from torch.hub"""
        model_info = self.model_dict.get(model_name, None)
        if not model_info:
            raise Exception('Unrecognized backbone model type')
        return load_model(model_info=model_info,
                          pretrained=True)

class NlstModel(nn.Module):
    """Model class for binary predction on NLST data"""
    def __init__(self,
                 backbone,
                 network_specification: List[int],
                 use_leaky_relu: bool = False):
        """Constructor"""
        super().__init__()
        self.backbone = backbone
        self.get_number_backbone_params()
        self.fully_connected_layers = create_network(network_specification)
        self.choose_activations(use_leaky_relu)
        self.dropout = nn.Dropout(p=.5)

    def choose_activations(self, use_leaky_relu):
        """Model will use ReLU or leaky ReLU depending on input"""
        if use_leaky_relu:
            self.change_backbone_to_leaky_relu()
            self.activation = F.leaky_relu
        else:
            self.activation = F.relu

    def change_backbone_to_leaky_relu(self):
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

    def to_cuda(self):
        """Sends the linear layers to cuda

        Only called if cuda device is available
        """
        if not torch.cuda.is_available():
            raise Exception('CUDA device not available')
        for layer in self.fully_connected_layers:
            layer.cuda()

    def apply_fully_connected_layers(self, input_tensor):
        """Applies all of the linear layers to the input"""
        if torch.cuda.is_available():
            #print('cuda is available')
            self.to_cuda()
        for layer in self.fully_connected_layers[:-1]:
            input_tensor = forward_through(linear_layer=layer,
                                activation=self.activation,
                                input_tensor=input_tensor,
                                dropout=self.dropout)
            #print(f"intermediate shape {input_tensor.shape}")
        output_layer = self.fully_connected_layers[-1]
        #print(f"output layer: {output_layer}")
        output_tensor = forward_through(linear_layer=output_layer,
                                        activation=None,
                                        input_tensor=input_tensor,
                                        dropout=None)
        return output_tensor

    def forward(self, input_tensor, clinical_info):
        """Function where the model predicts classes"""
        image_feature_map = self.backbone(input_tensor)
        combined_feature_tensor = torch.cat((clinical_info, image_feature_map),
                                            dim=1)

        output_tensor = self.apply_fully_connected_layers(
                input_tensor=combined_feature_tensor
                )

        return output_tensor

class ModelConstructor():
    """Class that build NLstModels with different specifications

    One can change the backbone, the netowrk specification, and the activations
    used
    """

    def __init__(self,
                 backbone_getter=BackboneGetter(known_models)
                 ):
        """Constructor"""
        self.backbone_getter = backbone_getter

    def construct_model(self,
                        backbone_name,
                        *nodes_per_layer,
                        use_leaky_relu=False):
        """Creates an NlstModel instance with the specifications provided"""
        network_specification = list(nodes_per_layer)
        backbone = self.backbone_getter(model_name=backbone_name)
        model = NlstModel(
                 backbone=backbone,
                 network_specification=network_specification,
                 use_leaky_relu=use_leaky_relu)
        return model



if __name__ == '__main__':
    print('No tests needed right now')
