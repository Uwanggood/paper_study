import sys

import cv2
import torch
import pandas as pd
from torch.distributions import transforms
from torch.utils.data import Dataset, DataLoader
from glob import glob
from torchvision.datasets.folder import find_classes
from torchvision import transforms
from torchvision.transforms import TrivialAugmentWide
from PIL import Image

import utils

if sys.platform == "win32":
    delimiter = "\\"
else:
    delimiter = "/"

__all__ = [
    "ImageDataset",
    # "PrefetchGenerator", "PrefetchDataLoader",
    "CPUPrefetcher", "CUDAPrefetcher",
]

# Image formats supported by the image processing library
IMG_EXTENSIONS = ("jpg", "jpeg", "png", "ppm", "bmp", "pgm", "tif", "tiff", "webp")


class ImageDataset(Dataset):
    def __init__(self,
                 image_dir: str,
                 image_target_dir: str,
                 image_size: int,
                 # Dataset loading method, the training data set is for data enancement, and the verification data set is not for data enhancement.
                 # and the verification data set  is not for data enhancement.
                 mode: str
                 ) -> None:
        super(ImageDataset, self).__init__()
        self.image_file_paths = glob(f"{image_dir}/*/*")
        self.class_to_idx = pd.read_csv(image_target_dir, index_col=0).to_dict()['name']
        self.image_size = image_size
        self.mode = mode
        self.delimeter = delimiter
        mode = str(self.mode).lower()
        if mode == "train":
            # Use Pytorch's own data enhancement to enlarge and enhance data
            self.pre_transform = transforms.Compose([
                transforms.RandomResizedCrop(self.image_size),
                TrivialAugmentWide(),
                transforms.RandomRotation([0, 270]),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
            ])
        elif mode == "valid" or mode == "test":
            # Use Pytorch's own data enhancement to enlarge and enhance data
            self.pre_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop([self.image_size, self.image_size]),
            ])
        else:
            raise ValueError("Unsupported data read type. Please use `Train` or `Valid` or `Test`")

        self.post_transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, batch_index: int) -> [torch.Tensor, int]:
        image_dir, image_name = self.image_file_paths[batch_index].split(self.delimeter)[-2:]
        # Read a batch of image data
        if image_name.split(".")[-1].lower() in IMG_EXTENSIONS:
            image = cv2.imread(self.image_file_paths[batch_index])
            target = self.class_to_idx[image_name]
        else:
            raise ValueError(
                f"Unsupported image extensions, Only support `{IMG_EXTENSIONS}`, please check the image file extensions.")

        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # OpenCV convert PIL
        image = Image.fromarray(image)

        # Data preprocess
        image = self.pre_transform(image)

        # Convert image data into Tensor stream format (Pytorch).
        # Note: The range of input and output is between [0, 1]
        tensor = utils.image_to_tensor(image, False, False)

        # Data postprocess
        tensor = self.post_transform(tensor)

        return {"image": tensor, "target": target}

    def __len__(self) -> int:
        return len(self.image_file_paths)


class CUDAPrefetcher:
    """
    Use the CUDA side to acclerate data reading

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)


class CPUPrefetcher:
    """Use the CPU side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader) -> None:
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)