import torch.nn as nn
from torchvision.models import vgg16

class VGG(nn.Module):
    """  """
    def __init__(self, pretrained=False):
        super().__init__()
        vgg = vgg16(pretrained=pretrained)
        self.feature_extractor = vgg.features
        
        # vgg_no_pooling = list()
        # for vgg_layer in list(vgg.features.children()):
        #     if type(vgg_layer) == torch.nn.modules.pooling.MaxPool2d:
        #         continue
        #     else:
        #         vgg_no_pooling.append(vgg_layer)
        
        layers = [nn.AdaptiveAvgPool2d((1, 1))]
        self.pool = nn.Sequential(*layers)
        
        layers = [nn.Linear(in_features=512, out_features=256, bias=True)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Linear(in_features=256, out_features=1, bias=True)]
        layers += [nn.Sigmoid()]
        self.tail = nn.Sequential(*layers)
        
    def forward(self, img):
        x = self.feature_extractor(img)
        x = self.pool(x)
        return self.tail(x.view(-1, 512))
