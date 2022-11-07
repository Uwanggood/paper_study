import torch
import model
import config
from utils import load_state_dict
from torch import nn


def build_model() -> nn.Module:  # nn.Module은 파이토치에서 제공하는 모듈이다.
    dict = model.__dict__[config.model_arch_name]
    if dict is None:
        raise ValueError("model.__dict__[config.model_arch_name] is not defined.")

    vgg_model = dict(num_classes=config.model_num_classes)
    vgg_model = vgg_model.to(device=config.device, memory_format=torch.channels_last)
    return vgg_model


def main() -> None:  # 화살표는 함수리턴에 대한 주석이다.(function annoation)
    # Initialize the model
    vgg_model = build_model()
    print(f"Build {config.model_arch_name.upper()} model successfully.")

    # Load model weights
    # 기존 소스는 weight를 불러와서 이용했지만 weight가 없으므로 torch.hub를 사용한다.
    vgg_model, _, _, _, _, _ = load_state_dict(vgg_model, config.model_weights_path)
    vgg_model.eval()

    test_prefetcher = model.load_dataset()

if __name__ == "__main__":
    main()
