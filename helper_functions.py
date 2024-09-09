"""This module contains all the functions which are used in multiple notebooks"""
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from fastai.vision.all import *
import argparse
import paths as p
import triplet_loss as tl
from tqdm import tqdm
from PIL import Image

def check_gpu():
    """ Check the enviroment and torch version """

    print("CUDA_LAUNCH_BLOCKING =", os.getenv('CUDA_LAUNCH_BLOCKING'))
    print("TORCH_USE_CUDA_DSA =", os.getenv('TORCH_USE_CUDA_DSA'))
    print("torch " + str(torch.__version__))
    print("torch.version.cuda " + str(torch.version.cuda))
    print("torch.backends.cudnn.version() " + str(torch.backends.cudnn.version()))
    print("Cuda is available: " + str(torch.cuda.is_available()))
    print("torch.backends.cudnn.enabled " + str(torch.backends.cudnn.enabled))


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for model training")

    parser.add_argument("-m", "--model", type=str,
                        help="model to train")
    parser.add_argument("-d", "--data_mode", type=str,
                        choices=['sample', 'full_data'],
                        help="load sample data or full dataset")
    parser.add_argument("-bs", "--batch_size", type=int,
                        help="number of batch size for data loader")
    parser.add_argument("-tn", "--tune_no", type=int,
                        help="number tuning rounds to perform")
    parser.add_argument("-lf", "--loss_func", type=str,
                        choices=["BaseLoss", "CrossEntropyLossFlat", "FocalLossFlat",
                        "BCEWithLogitsLossFlat", "L1LossFlat", "LabelSmoothingCrossEntropy",
                        "LabelSmoothingCrossEntropyFlat"],
                        help="loss function")
    parser.add_argument("-of", "--opt_func", type=str,
                        choices=["Adam", "AdamW", "FusedAdamW"],
                        help="optimization function")
    parser.add_argument("-lr", "--learning_rate", type=int,
                        help="learning rate")
    # parser.add_argument("--init", type=str,
    #                     choices=["kaiming_normal_"],
    #                     # HE & Xavier
    #                     help="initialization function")

    return parser.parse_args()


def load_data(data_type):
    """ Loads data for training or finetuning.
    Args:
    data_type: str
    Takes arguments "sample" or "full_data".
    Variable to asjust batch size.

    Returns:
    dls: data loader object
    Return data in data loader object. """

    print(p.data_train_sample)

    if data_type == 'sample':
        dls = ImageDataLoaders.from_folder(p.data_train_sample, train="Train", valid="Validation",
                                           item_tfms=Resize(224), bs=4, num_workers=0, drop_last=True)
    elif data_type == 'full_data':
        dls = ImageDataLoaders.from_folder(p.data_full, train="Train", valid="Validation",
                                           item_tfms=Resize(224), bs=24, num_workers=2, drop_last=True)
    else:
        raise ValueError("Invalid data type. Choose 'sample' or 'full_data'.")
    return dls


def augment_data(dls, n_classes):
    "Apply augmentation and debug the DataLoader for invalid targets"
    try:
        dls.train.after_item = Pipeline([ToTensor(), RandomResizedCrop(224, min_scale=0.5, ratio=(0.75, 1.33))])
        dls.train.after_batch = Pipeline([
            IntToFloatTensor(),
            Flip(p=0.5),
            Brightness(max_lighting=0.2, p=1.0)
        ])

        # Check target labels in the DataLoader
        unique_targets = set()
        for i, batch in enumerate(dls.train):
            inputs, targets = batch

            # Debug: Print shapes and types of inputs and targets
            # print(f"Batch {i}:")
            # print(f"  Inputs shape: {inputs.shape}, type: {type(inputs)}")
            # print(f"  Targets shape: {targets.shape}, type: {type(targets)}")

            unique_targets.update(targets.tolist())

            # Debug: Check the data range and values
            if inputs.ndim != 4 or inputs.shape[1] != 3:
                print(f"Unexpected input dimensions: {inputs.shape}")
                raise ValueError(f"Unexpected input dimensions: {inputs.shape}")

            if targets.ndim != 1:
                print(f"Unexpected target dimensions: {targets.shape}")
                raise ValueError(f"Unexpected target dimensions: {targets.shape}")

            # Check if any target is out of range
            if (targets < 0).any() or (targets >= n_classes).any():
                print("Found invalid targets in the dataset!")
                print("Invalid targets (less than 0):", targets[targets < 0])
                print("Invalid targets (greater than or equal to n_classes):", targets[targets >= n_classes])
                return None

        print(f"Unique targets in the dataset: {sorted(unique_targets)}")
        if max(unique_targets) >= n_classes:
            raise ValueError(f"n_classes ({n_classes}) is less than the maximum target value ({max(unique_targets)}). Please check your dataset and n_classes value.")
        
        print("Data augmentation completed successfully.")
        return dls  # Return the DataLoaders object

    except Exception as e:
        print(f"An error occurred during augmentation: {e}")
        raise

def custom_train_model(train_loader, model, criterion, optimizer, device, epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = []
        for step, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            anchor_img, positive_img, negative_img = anchor_img.to(device), positive_img.to(device), negative_img.to(device)
            
            optimizer.zero_grad()
            anchor_out = model(anchor_img)
            positive_out = model(positive_img)
            negative_out = model(negative_img)
            
            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()
            
            running_loss.append(loss.item())
        print(f"Epoch {epoch+1}/{epochs}, Loss: {np.mean(running_loss):.4f}")


def save_and_visualize_batches(data_loader, save_dir, num_batches=1):
    os.makedirs(save_dir, exist_ok=True)
    
    for batch_idx, (anchor_imgs, positive_imgs, negative_imgs, labels) in enumerate(data_loader):
        if batch_idx >= num_batches:
            break
        
        # Create a new figure for each batch
        fig = plt.figure(figsize=(12, 12))
        
        # Manually create subplots for images
        for i in range(len(anchor_imgs)):
            ax = fig.add_subplot(4, 3, 3*i+1)
            ax.imshow(anchor_imgs[i].permute(1, 2, 0).numpy())
            ax.set_title(f'Anchor: {labels[i].item()}')
            ax.axis('off')

            ax = fig.add_subplot(4, 3, 3*i+2)
            ax.imshow(positive_imgs[i].permute(1, 2, 0).numpy())
            ax.set_title(f'Positive: {labels[i].item()}')
            ax.axis('off')

            ax = fig.add_subplot(4, 3, 3*i+3)
            ax.imshow(negative_imgs[i].permute(1, 2, 0).numpy())
            ax.set_title(f'Negative: {labels[i].item()}')
            ax.axis('off')
        
        # Save the figure
        plt.savefig(os.path.join(save_dir, f'batch_{batch_idx}.png'))
        plt.close(fig)
