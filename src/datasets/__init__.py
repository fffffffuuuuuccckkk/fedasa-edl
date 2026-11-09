from .mnist import MNIST, FashionMNIST
from .cifar import CIFAR10, CIFAR100
from .svhn import SVHN
from .stl import STL10
from .speech_commands import GSC
from .tiny_imagenet import TinyImageNet
from .gtsrb import GTSRB
from .miotcd import MIOTCD
from .utils import *
from .randaugment import RandAugment

__all__ = ('MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'SVHN', 'STL10','GSC','TinyImageNet','GTSRB','MIOTCD')
