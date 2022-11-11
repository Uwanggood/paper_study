from enum import Enum

import numpy as np
import torch
from torch import nn
from torchvision.transforms import functional as F_vision
from torchvision.models import vgg19_bn, VGG19_BN_Weights


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


def load_state_dict(
        model: nn.Module,
        model_weights_path: str = None,
        ema_model: nn.Module = None,
        start_epoch: int = None,
        best_acc1: float = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        load_mode: str = None,
) -> [nn.Module, nn.Module, str, int, float, torch.optim.Optimizer, torch.optim.lr_scheduler]:
    # Load model weights
    state_dict = torch.load(model_weights_path,
                            map_location=lambda storage, loc: storage) if model_weights_path is not None \
        else VGG19_BN_Weights.DEFAULT.get_state_dict(True)

    if load_mode == "resume":
        ...
    else:
        model_state_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if
                      k in model_state_dict.keys() and v.size() == model_state_dict[k].size()}
        model_state_dict.update(state_dict)
        model.load_state_dict(model_state_dict)

    return model, ema_model, start_epoch, best_acc1, optimizer, scheduler


def image_to_tensor(image: np.ndarray, range_norm: bool, half: bool) -> torch.Tensor:
    """
     Convert the image data type to float32


    Args:
    :param image (np.ndarray): The image data read by ''OpenCV.imread'', the data range is [0,255] or [0.1]
    :param range_norm (bool): Scale [0, 1] data to between [-1. 1]
    :param half(bool): Whether to convert torch.float32 similarly to torch.half type
    :return: torch.Tensor
    """
    # Convert the image data type to float32
    tensor = F_vision.to_tensor(image)
    # Scale [0, 1] data to between [-1, 1]
    if range_norm:
        tensor = tensor.mul(2.0).sub(1.0)  # 이 부분이 왜 -1, 1이 되는지 봐야함

    if half:
        tensor = tensor.half()

    return tensor


class AverageMeter(object):
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.2f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.2f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.2f}"
        else:
            raise ValueError(f"Invalid summary type {self.summary_type}")

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))


        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            results.append(correct_k.mul_(100.0 / batch_size))
        return results

