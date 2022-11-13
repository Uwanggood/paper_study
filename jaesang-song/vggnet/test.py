import torch
from torchvision.io import image

import time
import model
import config
from utils import load_state_dict, AverageMeter, Summary, ProgressMeter, accuracy
from torch import nn
from dataset import CUDAPrefetcher, ImageDataset, CPUPrefetcher
from torch.utils.data import DataLoader

def load_dataset() -> CUDAPrefetcher:
    test_dataset = ImageDataset(config.test_image_dir,config.test_image_target_dir, config.image_size, "Test")
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=False,
                                 num_workers=config.num_workers,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

    # Place all data on the preprocessing data loader
    test_prefetcher = CUDAPrefetcher(test_dataloader, config.device) if config.device == "cuda" \
        else CPUPrefetcher(test_dataloader)

    return test_prefetcher

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
    # 기존 소스는 weight를 불러와서 이용했지만 weight가 없으므로 torch.hub를 사용한다.D
    vgg_model, _, _, _, _, _ = load_state_dict(vgg_model)
    vgg_model.eval()

    test_prefetcher = load_dataset()

    # Calculate how many batches of data are in each Epoch
    batches = len(test_prefetcher)
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    acc1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    acc5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
    progress = ProgressMeter(batches, [batch_time, acc1, acc5], prefix="Test: ")

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initilalize the data loader and load first batch of data
    test_prefetcher.reset()
    batch_data = test_prefetcher.next()

    # Get the initialize test time
    end = time.time()

    with torch.no_grad():
        while batch_data is not None:
            # Transfer in-memory data to CUDA device to speed up training
            images = batch_data["image"].to(device=config.device, non_blocking=True)
            target = batch_data["target"]

            # Get batch size
            batch_size = images.size(0)

            # inference
            output = vgg_model(images)

            # measure accuracy and record loss
            acc = accuracy(output, target)
            acc1.update(acc, batch_size)

            # Calculate the time it takes to fully train a batch of data
            batch_time.update(time.time() - end)
            end = time.time()

            # Write the data during training to the training log file
            if batch_index % config.test_print_frequency == 0:
                progress.display(batch_index + 1)

            # Preload the next batch of data
            batch_data = test_prefetcher.next()

            # Add 1 to the number of data batches to ensure that the terminal prints data normally
            batch_index += 1

    # print metrics
    progress.display_summary()
if __name__ == "__main__":
    main()
