import os
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from tqdm import tqdm

class RAFDBDataset(Dataset):
    def __init__(self, dataset_path, custom_transform=None, train=True):
        
        self.train = train
        self.dataset_path = dataset_path
        self.custom_transform = custom_transform

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.images = []
        self.images_path = []
        self.labels = []

        self.sub_dataset_path = Path(self.dataset_path) / "DATASET" / ("train" if self.train else "test")

        self.classes = os.listdir(self.sub_dataset_path)
        self.class_names = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
        self.class_counts = [0 for _ in range(len(self.class_names))]

        for class_idx in tqdm(self.classes, desc="Loading RAF-DB Dataset", leave=False):
            class_folder = self.sub_dataset_path / class_idx
            class_files = os.listdir(class_folder)

            label_int = int(class_idx) - 1

            self.class_counts[label_int] = len(class_files)

            for file_name in tqdm(class_files, desc=f"Processing class {class_idx}", leave=False):
                file_path = class_folder / file_name

                label = torch.tensor(label_int, dtype=torch.long)
                
                image = Image.open(file_path).convert('RGB')
                #image = self.transform(image)

                self.images_path.append(file_path)
                self.labels.append(label)
                self.images.append(image)

        self.class_weights = [
            1.0 / count if count > 0 else 1.0 for count in self.class_counts
        ]


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.custom_transform:
            image = self.custom_transform(image)

        # Apply the default transform
        image = self.transform(image)

        return image, label