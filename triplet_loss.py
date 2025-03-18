from torch.utils.data import Dataset
import random
import os

from matplotlib import transforms
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torch.nn.functional as F

# L(A, P, N) = max(0, D(A, P) — D(A, N) + margin)

class TripletNet(nn.Module):
    def __init__(self, base_model, embedding_dim=128):
        super(TripletNet, self).__init__()
        self.backbone = base_model
        in_features = self.backbone.fc.in_features  # Get input features of last FC layer
        self.backbone.fc = nn.Linear(in_features, embedding_dim)  # Replace FC layer with embedding layer

    def forward(self, x):
        return F.normalize(self.backbone(x), p=2, dim=1)  # Normalize embeddings


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
        
        # Return 4 elements: anchor, positive, negative, and label
        return anchor_img, positive_img, negative_img, anchor_label

    
class TripletModel(nn.Module):
    def __init__(self, base_model, embedding_dim=128):
        super().__init__()
        self.base_model = base_model

        # Handle different architectures
        if hasattr(self.base_model, "fc"):  # ResNet case
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(in_features, embedding_dim)  # Fully connected layer
        elif hasattr(self.base_model, "classifier"):  # ConvNeXt case
            in_features = self.base_model.classifier[-1].in_features
            self.base_model.classifier = nn.Sequential(nn.Linear(in_features, embedding_dim))
        else:
            raise ValueError(f"Unsupported model type: {type(self.base_model)}. Check model structure.")

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Ensure pooling before Linear

    def forward(self, x1, x2=None, x3=None):
        """Processes triplet inputs and returns embeddings"""
        if x2 is None and x3 is None:
            return self.get_embedding(x1)

        out1 = self.get_embedding(x1)
        out2 = self.get_embedding(x2)
        out3 = self.get_embedding(x3)
        return out1, out2, out3

    def get_embedding(self, x):
        """Extracts feature embeddings"""
        x = self.base_model.features(x)  # Extract feature maps (works for ResNet & ConvNeXt)
        x = self.global_pool(x)  # Apply Global Average Pooling (B, C, 1, 1)
        x = torch.flatten(x, 1)  # Flatten before passing to Linear (B, C)
        x = self.base_model.classifier(x)  # Fully connected layer
        return F.normalize(x, p=2, dim=1)  # Normalize embeddings


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def update_margin(self, new_margin):
        """Update the margin dynamically"""
        self.margin = new_margin

    def calc_euclidean(self, x1, x2, squared=True):
        """Computes Euclidean distance or squared Euclidean distance"""
        distance = (x1 - x2).pow(2).sum(1)
        if not squared:
            distance = torch.sqrt(distance + 1e-8)  # Adding epsilon for numerical stability
        return distance

    def get_hardest_triplets(self, anchor, positive, negative, model):
        """Find hardest positives & negatives for triplet loss."""
        with torch.no_grad():
            # Ensure that `model` is only used on images, not embeddings
            if anchor.dim() == 4:  # Only pass raw images (B, C, H, W) to `model`
                anchor_emb = model(anchor)
                pos_emb = model(positive)
                neg_emb = model(negative)
            else:
                anchor_emb, pos_emb, neg_emb = anchor, positive, negative  # Already processed embeddings

            pos_dist = self.calc_euclidean(anchor_emb, pos_emb)
            neg_dist = self.calc_euclidean(anchor_emb, neg_emb)

            hardest_positive = positive[pos_dist.argmax()]
            hardest_negative = negative[neg_dist.argmin()]

        return anchor_emb, hardest_positive, hardest_negative

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, model) -> torch.Tensor:
        """Computes triplet loss with hard triplet selection"""
        anchor, hardest_positive, hardest_negative = self.get_hardest_triplets(anchor, positive, negative, model)

        # Compute triplet loss using hardest triplets
        distance_positive = self.calc_euclidean(anchor, hardest_positive)
        distance_negative = self.calc_euclidean(anchor, hardest_negative)

        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()
   
class WrappedTripletLoss(nn.Module):
    def __init__(self, model, margin=1.0):
        super().__init__()
        self.triplet_loss = TripletLoss(margin=margin)
        self.model = model
    
    def forward(self, preds, target):
        # Unpack predictions (anchor, positive, negative) from preds
        anchor, positive, negative = preds
        # Pass anchor, positive, and negative to the triplet loss
        return self.triplet_loss(anchor, positive, negative, self.model)