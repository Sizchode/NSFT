import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch


class FER2013DataLoader:
    def __init__(self, csv_file):
        self.clip_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load the CSV file
        df = pd.read_csv(csv_file)

        # Split into train and test datasets
        train_df = df[df['Usage'] == 'Training']
        test_df = df[df['Usage'].isin(['PublicTest', 'PrivateTest'])]

        self.train_dataset = FER2013Dataset(train_df, transform=self.clip_transform)
        self.test_dataset = FER2013Dataset(test_df, transform=self.clip_transform)

        # FER2013 has 7 classes (0-6)
        self.class_to_idx = {str(i): i for i in range(7)}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Emotion mapping for reference
        self.emotion_mapping = {
            0: 'Angry',
            1: 'Disgust',
            2: 'Fear',
            3: 'Happy',
            4: 'Sad',
            5: 'Surprise',
            6: 'Neutral'
        }

        print("Class to Index Mapping:", self.class_to_idx)
        print("Emotion Mapping:", self.emotion_mapping)
        print(f"Total training samples loaded: {len(self.train_dataset)}")
        print(f"Total test samples loaded: {len(self.test_dataset)}")


class FER2013Dataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        # The 'pixels' column in FER2013 contains space-separated pixel values
        pixels = np.array(row['pixels'].split(' '), dtype=np.uint8)

        # Reshape to 48x48 grayscale image
        image = pixels.reshape(48, 48)

        # Convert to PIL Image
        image = Image.fromarray(image)

        image = image.convert('RGB')

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        label = int(row['emotion'])

        return image, str(label)  # Return label as string for class_to_idx mapping