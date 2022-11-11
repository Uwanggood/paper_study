import torch

model_arch_name = "vgg19_bn"
model_arch_url = {"vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
                  "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
                  "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
                  "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
                  "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
                  "vgg13": "https://download.pytorch.org/models/vgg13-c768596a.pth",
                  "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
                  "vgg11": "https://download.pytorch.org/models/vgg11-bbd30ac9.pth"}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_num_classes = 1000
model_weights_path = None
