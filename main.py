import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import requests
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchsummary import summary
from datasets import build_dataset
from distutils.util import strtobool
from tqdm import tqdm
import medmnist
from medmnist import INFO, Evaluator
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import natten
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize


def download_checkpoint(url, path):
    print(f"Downloading checkpoint from {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
    print(f"Checkpoint downloaded and saved to {path}")