from .ResNet import *
from .ResNets import *
from .VGG import *
from .VGG_LTH import *
from .swin import *

model_dict = {
    "resnet18": resnet18,
    "resnet50": resnet50,
    "resnet20s": resnet20s,
    "resnet44s": resnet44s,
    "resnet56s": resnet56s,
    "vgg16_bn": vgg16_bn,
    "vgg16_bn_lth": vgg16_bn_lth,
    "swin_t": swin_t,
    "resnet18extractor": ResNet18Extractor,
    "vgg19_bn": vgg19_bn,
}