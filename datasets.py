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


class PADatasetDownloader:
    def __init__(self, root_dir='data',
                 dataset_url='https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/zr7vgbcyr2-1.zip'):
        self.root_dir = root_dir
        self.dataset_url = dataset_url
        self.dataset_zip_path = os.path.join(self.root_dir, 'zr7vgbcyr2-1.zip')
        self.dataset_extracted_dir = self.root_dir
        self.source_images_dirs = [os.path.join(self.root_dir, 'images', f'imgs_part_{i}') for i in range(1, 4)]
        self.organized_images_dir = os.path.join(self.root_dir, 'PAD-Dataset')
        self.metadata_file_path = os.path.join(self.root_dir, 'metadata.csv')

    def download_dataset(self):
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

        if not os.path.exists(self.dataset_zip_path):
            print(f"Downloading dataset from {self.dataset_url}...")
            with requests.get(self.dataset_url, stream=True) as r:
                r.raise_for_status()
                with open(self.dataset_zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print("Download complete.")

    def extract_dataset(self):
        if not os.path.exists(os.path.join(self.root_dir, 'images')):
            print("Extracting main dataset...")
            with ZipFile(self.dataset_zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.root_dir)
            print("Main extraction complete.")

    def extract_inner_datasets(self):
        for i, source_images_dir in enumerate(self.source_images_dirs, start=1):
            inner_zip_path = os.path.join(self.root_dir, f'images/imgs_part_{i}.zip')
            if not os.path.exists(source_images_dir):
                print(f"Extracting {inner_zip_path}...")
                with ZipFile(inner_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(os.path.dirname(source_images_dir))
                print(f"Extraction of {inner_zip_path} complete.")

    def organize_images(self):
        if os.path.exists(self.organized_images_dir):
            print("Images are already organized.")
            return

        if not os.path.exists(self.metadata_file_path):
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_file_path}")

        metadata = pd.read_csv(self.metadata_file_path)

        os.makedirs(self.organized_images_dir, exist_ok=True)

        diagnostic_labels = metadata['diagnostic'].unique()

        for label in diagnostic_labels:
            os.makedirs(os.path.join(self.organized_images_dir, label), exist_ok=True)

        for _, row in metadata.iterrows():
            img_id = row['img_id']
            diagnostic = row['diagnostic']

            for source_dir in self.source_images_dirs:
                source_path = os.path.join(source_dir, img_id)
                if os.path.exists(source_path):
                    destination_path = os.path.join(self.organized_images_dir, diagnostic, img_id)
                    shutil.move(source_path, destination_path)
                    break

        print("Images moved successfully.")

    def get_dataset(self):
        if os.path.exists(self.organized_images_dir):
            print("Dataset already exists. Returning the root directory.")
            return self.organized_images_dir
        else:
            self.download_dataset()
            self.extract_dataset()
            self.extract_inner_datasets()
            self.organize_images()
            return self.organized_images_dir


class FetalDatasetDownloader:
    def __init__(self, root_dir='data', dataset_url='https://zenodo.org/records/3904280/files/FETAL_PLANES_ZENODO.zip'):
        self.root_dir = root_dir
        self.dataset_url = dataset_url
        self.dataset_zip_path = os.path.join(self.root_dir, 'FETAL_PLANES_ZENODO.zip')
        self.dataset_extracted_dir = self.root_dir
        self.organized_images_dir = os.path.join(self.root_dir, 'Fetal-Dataset')
        self.excel_file_path = os.path.join(self.root_dir, 'FETAL_PLANES_DB_data.xlsx')
        self.source_images_dir = os.path.join(self.root_dir, 'Images')

    def download_dataset(self):
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

        if not os.path.exists(self.dataset_zip_path):
            print(f"Downloading dataset from {self.dataset_url}...")
            with requests.get(self.dataset_url, stream=True) as r:
                r.raise_for_status()
                with open(self.dataset_zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print("Download complete.")

    def extract_dataset(self):
        if not os.path.exists(self.excel_file_path) or not os.path.exists(self.source_images_dir):
            print("Extracting dataset...")
            with ZipFile(self.dataset_zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.root_dir)
            print("Extraction complete.")

    def organize_images(self):
        if os.path.exists(self.organized_images_dir):
            print("Images are already organized.")
            return

        if not os.path.exists(self.excel_file_path):
            raise FileNotFoundError(f"Excel file not found at {self.excel_file_path}")

        df = pd.read_excel(self.excel_file_path)

        os.makedirs(self.organized_images_dir, exist_ok=True)

        plane_labels = df['Plane'].unique()

        for label in plane_labels:
            os.makedirs(os.path.join(self.organized_images_dir, str(label)), exist_ok=True)

        for _, row in df.iterrows():
            img_id = row['Image_name']
            plane = row['Plane']
            source_path = os.path.join(self.source_images_dir, f'{img_id}.png')
            destination_path = os.path.join(self.organized_images_dir, str(plane), f'{img_id}.png')

            if os.path.exists(source_path):
                shutil.move(source_path, destination_path)

        print("Images moved successfully.")

    def get_dataset(self):
        if os.path.exists(self.organized_images_dir):
            print("Dataset already exists. Returning the root directory.")
            return self.organized_images_dir
        else:
            self.download_dataset()
            self.extract_dataset()
            self.organize_images()
            return self.organized_images_di


