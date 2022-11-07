import sys

from torch.distributions import transforms
from torch.utils.data import Dataset, DataLoader
from glob import glob
from torchvision.datasets.folder import find_classes
from torchvision import transforms
from torchvision.transforms import TrivialAugmentWide

if sys.platform == "win32":
    delimiter = "\\"
else:
    delimiter = "/"


class ImageDataset(Dataset):
    def __init__(self,
                 image_dir: str,
                 image_size: int,
                 # Dataset loading method, the training data set is for data enancement, and the verification data set is not for data enhancement.
                 # and the verification data set  is not for data enhancement.
                 mode:str
                 ) -> None:
        super(ImageDataset, self).__init__()
        self.image_file_paths = glob(f"{image_dir}/*/*")

        _, self.class_to_idx = find_classes(image_dir)
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
            ... # see you
