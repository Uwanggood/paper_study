import torch
import model
import config
from torch import nn


def build_model() -> nn.Module: #nn.Module은 파이토치에서 제공하는 모듈이다.
    vgg_model = model.__dict__[config.model_arch_name](num_classes=config.model_num_classes)
    vgg_model = vgg_model.to(device=config.device, memory_format=torch.channels_last)


def main() -> None: #화살표는 함수리턴에 대한 주석이다.(function annoation)
    # Initialize the model
    vgg_model = build_model()


if __name__ == "__main__":
    main()