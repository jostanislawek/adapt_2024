import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict

# L(A, P, N) = max(0, D(A, P) — D(A, N) + margin)

class TripletModel(nn.Module):
    def __init__(self, base_model, embedding_dim=128):
        super().__init__()
        self.embedding = base_model

        # Handle different architectures (ResNet, ConvNeXt)
        if hasattr(self.embedding, "fc"):  # ResNet case
            in_features = self.embedding.fc.in_features
            self.embedding.fc = nn.Linear(in_features, embedding_dim)
        elif hasattr(self.embedding, "classifier"):  # ConvNeXt case
            in_features = self.embedding.classifier[-1].in_features
            self.embedding.classifier = nn.Sequential(nn.Linear(in_features, embedding_dim))
        else:
            raise ValueError(f"Unsupported model type: {type(self.embedding)}. Check model structure.")

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

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
        x = self.embedding.features(x)
        x = self.global_pool(x)  # Global Average Pooling
        x = torch.flatten(x, 1)  # Flatten before passing to Linear
        x = self.embedding.classifier(x)  # Fully connected layer
        return F.normalize(x, p=2, dim=1)  # Normalize embeddings

class TripletDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.images = []
        self.labels = []
        self.class_counts = defaultdict(int)

        print("Class folders found:", os.listdir(root_dir))
        
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            print("cls_name: ", cls_name)

            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)

                if os.path.isfile(img_path):
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[cls_name])
                    self.class_counts[cls_name] += 1
                else:
                    print(f"Skipping non-image file or directory: {img_path}")

        # Print dataset size
        print(f"Loaded {len(self.images)} images from {len(self.classes)} classes.")
        print("Images per class:")
        for cls_name in self.classes:
            print(f"  {cls_name}: {self.class_counts[cls_name]} images")
    
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

    def get_dataset_info(self):

        return {
                "num_images": len(self.images),
                "num_classes": len(self.class_to_idx),
                "labels": self.labels,
                "class_to_idx": self.class_to_idx
            }
        



class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def update_margin(self, new_margin):
        self.margin = new_margin

    def calc_euclidean(self, x1, x2, squared=True):
        distance = (x1 - x2).pow(2).sum(1)
        if not squared:
            distance = torch.sqrt(distance + 1e-8)
        return distance

    # def get_hardest_triplets(self, anchor, positive, negative, model):
    #     """Find hardest positives & negatives for triplet loss."""
    #     with torch.no_grad():
    #         # Ensure model is only used on images, not embeddings
    #         if anchor.dim() == 4:
    #             anchor_emb = model(anchor)
    #             pos_emb = model(positive)
    #             neg_emb = model(negative)
    #         else:
    #             anchor_emb, pos_emb, neg_emb = anchor, positive, negative

    #         pos_dist = self.calc_euclidean(anchor_emb, pos_emb)
    #         neg_dist = self.calc_euclidean(anchor_emb, neg_emb)

    #         hardest_positive = positive[pos_dist.argmax()]
    #         hardest_negative = negative[neg_dist.argmin()]

    #     return anchor_emb, hardest_positive, hardest_negative

    def get_semi_hard_triplets(self, anchor, positive, negative, model):
        with torch.no_grad():
            # Convert images to embeddings if needed
            if anchor.dim() == 4:
                anchor_emb = model(anchor)
                pos_emb = model(positive)
                neg_emb = model(negative)
            else:
                anchor_emb, pos_emb, neg_emb = anchor, positive, negative

            # Compute distances
            pos_dist = self.calc_euclidean(anchor_emb, pos_emb)
            neg_dist = self.calc_euclidean(anchor_emb, neg_emb)

            # Identify semi-hard negatives:
            # where pos_dist < neg_dist < pos_dist + margin
            semi_hard_mask = (pos_dist < neg_dist) & (neg_dist < pos_dist + self.margin)

            valid_indices = semi_hard_mask.nonzero(as_tuple=True)[0]

            hardest_positive = positive[pos_dist.argmax()]

            if len(valid_indices) > 0:
                random_idx = random.randint(0, len(valid_indices) - 1)
                semi_hard_negative = negative[valid_indices[random_idx]]
            else:
                semi_hard_negative = negative[neg_dist.argmin()]

        return anchor_emb, hardest_positive, semi_hard_negative

    def forward(self, anchor, positive, negative, model):
        """Computes triplet loss with hard triplet selection"""
        anchor, hardest_positive, hardest_negative = self.get_semi_hard_triplets(anchor, positive, negative, model)

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