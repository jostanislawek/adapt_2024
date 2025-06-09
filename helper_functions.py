"""This module contains all the functions which are used in multiple notebooks"""
import argparse
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from fastai.vision.all import *
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms
from tqdm import tqdm

import paths as p


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
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--margin', type=float, default=1.0, help='Triplet loss margin')
    parser.add_argument('--initial_margin', type=float, default=0.5, help='Initial margin value for triplet loss')
    parser.add_argument('--final_margin', type=float, default=1.5, help='Final margin value for triplet loss')

    # parser.add_argument("--resume", type=bool,
    #                     help="Resume training from checkpoint")
    parser.add_argument("--resume", action="store_true")
    # parser.add_argument("--init", type=str,
    #                     choices=["kaiming_normal_"],
    #                     # HE & Xavier
    #                     help="initialization function")
    parser.add_argument("--note", type=str, default="", help="Optional note.")

    return parser.parse_args()


def parent_label_func(filepath):
    return parent_label(filepath)


# Creates folders and subfolders for train or validate and the classes
def create_folder_structure(args, fold_index=None):
    """
    Creates a structured directory for the model, ensuring continuation when resuming.
    
    If using cross-validation, stores intermediate folds in "crossval_results/"
    and only keeps the last fold in the main model directory.
    """

    main_path = args.model
    os.makedirs(main_path, exist_ok=True)

    # Define base folder name
    base_folder_name = f"model_{args.model}_data_mode_{args.data_mode}_batch_size_{args.batch_size}"

    # Default tuning values
    tune_no = args.tune_no
    resume = args.resume

    # Find previous models
    existing_folders = [f for f in os.listdir(main_path) if base_folder_name in f]
    existing_folders = sorted(
        existing_folders, 
        key=lambda x: int(re.search(r"tune_no_(\d+)", x).group(1)) if re.search(r"tune_no_(\d+)", x) else 0
    )

    if resume and existing_folders:
        latest_folder = existing_folders[-1]
        match = re.search(r"tune_no_(\d+)", latest_folder)
        if match:
            previous_tune_no = int(match.group(1))
            tune_no += previous_tune_no  # Accumulate total tuning count
    else:
        tune_no = args.tune_no

    # Keep only the last fold in the main directory
    if fold_index is not None and fold_index < 4:  # First 4 folds go to "crossval_results/"
        folder_name = f"{base_folder_name}_tune_no_{tune_no}_resume_{resume}/crossval_results/fold_{fold_index+1}"
    else:  # Last fold goes in the main directory
        folder_name = f"{base_folder_name}_tune_no_{tune_no}_resume_{resume}"

    subfolder_path = os.path.join(main_path, folder_name)
    os.makedirs(subfolder_path, exist_ok=True)

    print(f"Created/Found model folder: {subfolder_path}")

    # Save Notes
    if args.note:
        note_path = os.path.join(subfolder_path, "notes.txt")
        with open(note_path, "a" if os.path.exists(note_path) else "w") as f:
            f.write(f"\n[Resume] {args.note}" if os.path.exists(note_path) else args.note)
        print(f"Note saved in {note_path}")

    return subfolder_path


def load_data(data_type):
    """ Loads data for training or finetuning.
    Args:
    data_type: str
    Takes arguments "sample" or "full_data".
    Variable to asjust batch size.

    Returns:
    dls: data loader object
    Return data in data loader object. """

    if data_type == 'sample':
        dls = ImageDataLoaders.from_folder(p.data_train_sample, train="Train", valid="Test",
                                           item_tfms=Resize(224), bs=4, num_workers=0, drop_last=False)
    elif data_type == 'full_data':
        dls = ImageDataLoaders.from_folder(p.data_full, train="Train", valid="Test",
                                           item_tfms=Resize(224), bs=4, num_workers=2, drop_last=False)
    else:
        raise ValueError("Invalid data type. Choose 'sample' or 'full_data'.")
    return dls


def load_data_crossval_stratified(args, data_mode, n_splits=5, fold_index=0):
    """
    Loads data for stratified cross-validation.

    Args:
        args: Command-line arguments containing data mode.
        data_mode (str): Either "sample" or "full_data".
        n_splits (int): Number of CV folds.
        fold_index (int): Fold index to use as validation.

    Returns:
        dict: Contains 'train_dls' and 'valid_dls'.
    """
    
    print(f"Loading data: {data_mode} (Stratified Cross-Validation: Fold {fold_index+1}/{n_splits})")

    if data_mode == 'sample':
        data_path = Path(p.data_train_sample)  # Convert to Path
        batch_size = 4
        print(data_path)
    elif data_mode == 'full_data':
        data_path = Path(p.data_full)  # Convert to Path
        batch_size = 24
        print(data_path)
    else:
        raise ValueError(f"Invalid data type: {data_mode}. Choose 'sample' or 'full_data'.")

    # Get all image files and labels
    all_files = get_image_files(data_path / "Train")  # Now works with Path
    all_labels = [parent_label(f) for f in all_files]  # Extract labels from folder names
    unique_labels = list(set(all_labels))

    # Convert labels to numeric values
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    numeric_labels = np.array([label_to_idx[label] for label in all_labels])

    # Create stratified cross-validation splits
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_idx, valid_idx = list(skf.split(all_files, numeric_labels))[fold_index]

    train_files, valid_files = np.array(all_files)[train_idx].tolist(), np.array(all_files)[valid_idx].tolist()

    # Define label function to extract labels
    def label_func(file_path):
        return parent_label(file_path)

    # Train & Validation DataLoaders
    train_dls = ImageDataLoaders.from_path_func(
        path=data_path, 
        fnames=train_files,  # Pass file names
        label_func=parent_label_func,
        item_tfms=Resize(224), 
        bs=batch_size, 
        num_workers=2, 
        drop_last=True
    )

    valid_dls = ImageDataLoaders.from_path_func(
        path=data_path, 
        fnames=valid_files,  # Pass file names
        label_func=parent_label_func,
        item_tfms=Resize(224), 
        bs=batch_size, 
        num_workers=2, 
        drop_last=False
    )

    print(f"Data Loaded: Train ({len(train_dls.train)}) | Valid ({len(valid_dls.valid)})")

    return {"train_dls": train_dls, "valid_dls": valid_dls}

def get_all_test_embeddings(dls, model):
    """ Extracts embeddings for all test images. """
    test_dl = dls.valid  # Get the test dataloader
    test_embeddings = []
    test_labels = []

    for batch in test_dl:
        images, labels = batch
        embeddings = model(images)  # Get embeddings using your model
        test_embeddings.append(embeddings)
        test_labels.append(labels)

    # Convert to numpy arrays
    test_embeddings = torch.cat(test_embeddings).detach().cpu().numpy()
    test_labels = torch.cat(test_labels).detach().cpu().numpy()

    return test_embeddings, test_labels

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

def save_and_visualize_batches(data_loader, save_dir, num_batches=1):
    os.makedirs(save_dir, exist_ok=True)
    
    for batch_idx, (anchor_imgs, positive_imgs, negative_imgs, labels) in enumerate(data_loader):
        if batch_idx >= num_batches:
            break
        
        num_imgs = len(anchor_imgs)
        # Create a figure with dynamic height based on the number of images
        fig = plt.figure(figsize=(12, 4 * num_imgs))
        
        # Manually create subplots for images, using a grid of num_imgs x 3
        for i in range(num_imgs):
            ax = fig.add_subplot(num_imgs, 3, 3 * i + 1)
            ax.imshow(anchor_imgs[i].permute(1, 2, 0).numpy())
            ax.set_title(f'Anchor: {labels[i].item()}')
            ax.axis('off')

            ax = fig.add_subplot(num_imgs, 3, 3 * i + 2)
            ax.imshow(positive_imgs[i].permute(1, 2, 0).numpy())
            ax.set_title(f'Positive: {labels[i].item()}')
            ax.axis('off')

            ax = fig.add_subplot(num_imgs, 3, 3 * i + 3)
            ax.imshow(negative_imgs[i].permute(1, 2, 0).numpy())
            ax.set_title(f'Negative: {labels[i].item()}')
            ax.axis('off')
        
        # Save the figure
        plt.savefig(os.path.join(save_dir, f'batch_{batch_idx}.png'))
        plt.close(fig)



def visualize_embeddings_tsne(embeddings, labels, save_path):
    """
    Visualizes embeddings using t-SNE and saves the plot.

    Args:
        embeddings (np.ndarray): Array of embeddings (shape: [N, D]).
        labels (np.ndarray): Array of labels corresponding to each embedding.
        save_path (str): Path where the plot will be saved.
    """
    # Reduce the embeddings to 2 dimensions using TSNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot the t-SNE results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.title("t-SNE Visualization of Embeddings")
    
    # Save the figure to the specified path
    plt.savefig(save_path)
    plt.close()

