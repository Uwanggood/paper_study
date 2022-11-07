from torch import nn
from typing import cast, Dict, List, Union
import torch
from torch import Tensor
from torch import nn

__all__ = [
    "VGG", "vgg19_bn",
]

vgg_cfgs: Dict[str, List[Union[str, int]]] = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _make_layers(vgg_cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: nn.Sequential[nn.Module] = nn.Sequential()  # :은 변수주석
    in_channels = 3
    for v in vgg_cfg:
        if v == "M":
            layers.append(nn.MaxPool2d((2, 2), (2, 2)))
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, (3, 3), (1, 1), (1, 1))
            if batch_norm:
                layers.append(conv2d)
                layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(True))
            else:
                layers.append(conv2d)
                layers.append(nn.ReLU(True))
            in_channels = v

    return layers


class VGG(nn.Module):
    def __init__(self, vgg_cfg: List[Union[str, int]], batch_norm: bool = False, num_classes: int = 1000) -> None:
        super(VGG, self).__init__()
        self.features = _make_layers(vgg_cfg, batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classfier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classfier(x)
        return x

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                #Xavier & He initialization 초기화(https://blog.naver.com/PostView.nhn?blogId=hongjg3229&logNo=221564537122)
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

def vgg19_bn(**kwargs) -> VGG:
    return VGG(vgg_cfgs["vgg19"], True, **kwargs)


def load_dataset():
    test_dataset = ImageDataset(config.test_image_dir, config.image_size, "Test")