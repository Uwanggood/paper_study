import torch

model_arch_name = "vgg19"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

