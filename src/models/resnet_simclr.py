import torch.nn as nn
import torchvision.models as models
from .resnet_hacks import modify_resnet_model


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, num_classes=10):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=num_classes),
                            "resnet50": models.resnet50(pretrained=False, num_classes=num_classes)}

        self.encoder = self._get_basemodel(base_model)
        
        # Lazy impelementation of cifar10
        # TODO: Fix this in future
        if num_classes == 10:
            self.encoder = modify_resnet_model(self.encoder)
        self.dim_mlp = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()

        self.linear = nn.Linear(self.dim_mlp, num_classes)
        self.freeze_feature = False

    def _get_basemodel(self, model_name):
        model = self.resnet_dict[model_name]
        return model

    def forward(self, x, return_features=False, specify_input_layer=None):
        # Only implements cases for finalembed
        if specify_input_layer:
            assert specify_input_layer == 'finalembed'
            return self.linear(x)

        intermediate = self.encoder(x)
        if self.freeze_feature:
            intermediate = intermediate.detach()
        out = self.linear(intermediate)
        if return_features:
            return out, intermediate
        return out
