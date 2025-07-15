import torch

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def get_default_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
