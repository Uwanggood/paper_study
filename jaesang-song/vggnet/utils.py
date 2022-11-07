import torch
import model
from config import model_arch_name, model_arch_url
from torch import nn


def load_model_url() -> str:
    return model_arch_url[model_arch_name]


def load_state_dict(
        model: nn.Module,
        model_weights_path: str,
        ema_model: nn.Module = None,
        start_epoch: int = None,
        best_acc1: float = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        load_mode: str = None,
) -> [nn.Module, nn.Module, str, int, float, torch.optim.Optimizer, torch.optim.lr_scheduler]:
    checkpoint = \
        torch.load(model_weights_path, map_location=lambda storage, loc: storage) if model_weights_path is not None \
            else torch.hub.load_state_dict_from_url(load_model_url())

    if load_mode == "resume":
        pass
    else:
        model_state_dict = model.state_dict()
        state_dict = {k: v for k, v in checkpoint.items() if
                      k in model_state_dict.keys() and v.size() == model_state_dict[k].size()}

        # Overwrite the model weights to the current model
        model_state_dict.update(state_dict)
        model.load_state_dict(model_state_dict)
    return model, ema_model, start_epoch, best_acc1, optimizer, scheduler  # model을 제외한 나머지부분은 resume일 경우에만 생성
