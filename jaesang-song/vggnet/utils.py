import torch
from torch import nn


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
    # Load model weights
    checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)

    if load_mode == "resume":
        ...
    else:
        model_state_dict = model.state_dict()
