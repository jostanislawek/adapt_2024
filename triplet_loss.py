from torch.utils.data import Dataset
import random
import os

from matplotlib import transforms
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# L(A, P, N) = max(0, D(A, P) — D(A, N) + margin)

class TripletDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.images = []
        self.labels = []

        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                self.images.append(os.path.join(cls_dir, img_name))
                self.labels.append(self.class_to_idx[cls_name])
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        anchor_path = self.images[idx]
        anchor_label = self.labels[idx]
        
        # Positive sample
        positive_indices = [i for i in range(len(self.labels)) if self.labels[i] == anchor_label and i != idx]
        positive_idx = random.choice(positive_indices)
        positive_path = self.images[positive_idx]
        
        # Negative sample
        negative_indices = [i for i in range(len(self.labels)) if self.labels[i] != anchor_label]
        negative_idx = random.choice(negative_indices)
        negative_path = self.images[negative_idx]
        
        anchor_img = Image.open(anchor_path).convert('RGB')
        positive_img = Image.open(positive_path).convert('RGB')
        negative_img = Image.open(negative_path).convert('RGB')
        
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        
        return anchor_img, positive_img, negative_img, anchor_label
        
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()


