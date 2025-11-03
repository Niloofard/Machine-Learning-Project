import os
import json
from torch.utils.data import DataLoader, random_split, Subset
import torch

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import medmnist
from medmnist import INFO, Evaluator
import requests
from zipfile import ZipFile
import pandas as pd
import shutil

# Set the random seed for reproducibility
seed = 42
torch.manual_seed(seed)


import os
import requests
from zipfile import ZipFile
import pandas as pd
import shutil

root_dir='data'
if not os.path.exists(root_dir):
            os.makedirs(root_dir)