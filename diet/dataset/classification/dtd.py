import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import random


class DTDDataLoader:
    def __init__(self, data_dir):
        self.clip_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Get the class names (folder names)
        self.classes = sorted([d for d in os.listdir(os.path.join(data_dir, 'images'))
                               if os.path.isdir(os.path.join(data_dir, 'images', d))])

        # Create class mappings
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for i, cls_name in enumerate(self.classes)}

        # Load the train-val-test splits
        train_files, val_files, test_files = self._load_splits(data_dir)

        # Create datasets
        self.train_dataset = DTDDataset(data_dir, train_files, self.class_to_idx, transform=self.clip_transform)
        self.val_dataset = DTDDataset(data_dir, val_files, self.class_to_idx, transform=self.clip_transform)
        self.test_dataset = DTDDataset(data_dir, test_files, self.class_to_idx, transform=self.clip_transform)

        print("Class to Index Mapping sample:",
              {k: self.class_to_idx[k] for k in list(self.class_to_idx.keys())[:5]} if self.classes else {})
        print(f"Total classes: {len(self.classes)}")
        print(f"Total training samples loaded: {len(self.train_dataset)}")
        print(f"Total validation samples loaded: {len(self.val_dataset)}")
        print(f"Total test samples loaded: {len(self.test_dataset)}")

    def _load_splits(self, data_dir):
        """Load the predefined train-val-test splits from DTD"""
        train_files = []
        val_files = []
        test_files = []

        # DTD provides 10 different train-val-test splits
        # We'll use split 1 (you could modify to use other splits or average results across splits)
        split_num = 1

        train_file = os.path.join(data_dir, 'labels', f'train{split_num}.txt')
        val_file = os.path.join(data_dir, 'labels', f'val{split_num}.txt')
        test_file = os.path.join(data_dir, 'labels', f'test{split_num}.txt')

        with open(train_file, 'r') as f:
            train_files = [line.strip() for line in f.readlines()]

        with open(val_file, 'r') as f:
            val_files = [line.strip() for line in f.readlines()]

        with open(test_file, 'r') as f:
            test_files = [line.strip() for line in f.readlines()]

        return train_files, val_files, test_files


class DTDDataset(Dataset):
    def __init__(self, root_dir, file_list, class_to_idx, transform=None):
        self.root_dir = root_dir
        self.file_list = file_list
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        # The file format in the list is like "category/image.jpg"
        category = img_name.split('/')[0]

        # Load the image
        img_path = os.path.join(self.root_dir, 'images', img_name)
        image = Image.open(img_path).convert('RGB')

        # Get class label
        label = self.class_to_idx[category]

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, str(label)  # Return label as string for class_to_idx mapping