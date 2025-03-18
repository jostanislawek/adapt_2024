import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

from torchvision import transforms
import torch
import inspect
from torch.utils.data import DataLoader
from fastai.vision.all import *
import helper_functions as hf
import model_helpers as mh
import paths as p
import triplet_loss as tl
from sklearn.model_selection import StratifiedKFold
import numpy as np
from pathlib import Path
from torch.utils.data import Subset
from fastai.callback.core import Callback
import matplotlib.pyplot as plt
import os
import mlflow
from triplet_loss import TripletModel

def main(args, fold_index):
    """Main function for training the Triplet Loss model with Cross-Validation."""

    model_folder_path = hf.create_folder_structure(args, fold_index)
    if not model_folder_path:
        raise ValueError("Failed to create model folder path.")
    print(f"Main function processing with folder path: {model_folder_path}")

    return model_folder_path


class ConstantMarginCallback(Callback):
    def before_epoch(self):
        # Always set the margin to the value from args.margin
        self.learn.loss_func.triplet_loss.update_margin(args.margin)
        print(f"Epoch {self.epoch}: Constant margin set to {args.margin}")


class LoggingCallback(Callback):
    def before_fit(self):
        self.epoch_losses = []
        self.epoch_metrics = []
        self.epoch_margins = []

    def after_epoch(self):
        # Record the training loss (or validation loss if desired)
        self.epoch_losses.append(self.learn.recorder.losses[-1].item())
        # Record a custom metric if provided (or you can compute one)
        # For example, if you have accuracy as a metric:
        if self.learn.recorder.metrics:
            self.epoch_metrics.append(self.learn.recorder.metrics[-1][0].item())
        # Log the current margin value if applicable
        current_margin = self.learn.loss_func.triplet_loss.margin
        self.epoch_margins.append(current_margin)
        print(f"Epoch {self.epoch}: Loss = {self.epoch_losses[-1]:.4f}, Margin = {current_margin:.4f}")

class FlexibleMarginCallback(Callback):
    def __init__(self, initial_margin, final_margin, total_epochs):
        self.initial_margin = initial_margin
        self.final_margin = final_margin
        self.total_epochs = total_epochs

    def before_epoch(self):
        # Calculate new margin value linearly over epochs
        new_margin = self.initial_margin + (self.final_margin - self.initial_margin) * (self.epoch / self.total_epochs)
        self.learn.loss_func.triplet_loss.update_margin(new_margin)
        print(f"Epoch {self.epoch}: Margin updated to {new_margin:.4f}")

class MLflowLoggerCallback(Callback):
    def before_fit(self):
        # Log parameters only once per run
        if not hasattr(self, 'params_logged'):
            mlflow.log_param("learning_rate", self.learn.opt.hypers[0]['lr'])
            mlflow.log_param("batch_size", self.learn.dls.train.bs)
            mlflow.log_param("epochs", self.learn.n_epoch)
            mlflow.log_param("model", str(self.learn.model.__class__.__name__))
            if hasattr(self.learn.loss_func.triplet_loss, 'margin'):
                mlflow.log_param("margin", self.learn.loss_func.triplet_loss.margin)
            self.params_logged = True  # Flag that parameters have been logged
            print("MLflow parameters logged.")

    def after_epoch(self):
        epoch = self.epoch
        if self.learn.recorder.losses:
            train_loss = self.learn.recorder.losses[-1].item()
            mlflow.log_metric("train_loss", train_loss, step=epoch)
        if self.learn.recorder.values:
            valid_loss = self.learn.recorder.values[-1][0]
            mlflow.log_metric("valid_loss", valid_loss, step=epoch)
            for i, m in enumerate(self.learn.recorder.values[-1][1:]):
                mlflow.log_metric(f"metric_{i}", m, step=epoch)
        print(f"Epoch {epoch}: metrics logged.")

def after_fit(self):
    example_input = torch.randn(1, 3, 224, 224).to("cuda")
    # Use get_embedding to obtain a single embedding output
    example_output = self.learn.model.get_embedding(example_input)
    signature = infer_signature(example_input.cpu().numpy(),
                                example_output.cpu().detach().numpy())
    mlflow.pytorch.log_model(self.learn.model, "model", input_example=example_input, signature=signature)
    print("Model logged to MLflow with input example and signature.")


if __name__ == "__main__":

    # Verify GPU is available
    hf.check_gpu()

    args = hf.parse_args()
    print(args)

    n_splits = 5  # Use 5-Fold Stratified Cross-Validation

    # Load the full dataset for cross-validation
    print("Loading dataset for Cross-Validation.")

    transforms = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load full dataset (no split yet)
    full_dataset = tl.TripletDataset(
        root_dir=p.data_train_sample_train, transform=transforms
    )

    # Get dataset labels
    all_labels = full_dataset.labels  # Assuming labels are stored

    # Convert labels to numeric values
    unique_labels = list(set(all_labels))
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    numeric_labels = np.array([label_to_idx[label] for label in all_labels])

    # Create stratified splits using your numeric labels
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, valid_idx) in enumerate(skf.split(full_dataset.images, numeric_labels)):
        print(f"\nTraining Fold {fold+1}/{n_splits}")

        # Create subsets for the training and validation sets
        train_dataset = Subset(full_dataset, train_idx)
        valid_dataset = Subset(full_dataset, valid_idx)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        dls = DataLoaders(train_loader, valid_loader)

        # Create model folder for this fold
        model_folder_path = hf.create_folder_structure(args, fold)

        # Initialize Feature Extractor (Base Model)
        print("Creating model.")
        base_model = mh.create_model_object(args.model)

        # Initialize Triplet Model with Base Model
        triplet_model = TripletModel(base_model, embedding_dim=128)

        # with mlflow.start_run():
        #     loss_func = tl.WrappedTripletLoss(model=model, margin=args.margin)
        #     learner = Learner(
        #         dls, 
        #         triplet_model, 
        #         loss_func=loss_func, 
        #         metrics=[], 
        #         cbs=[
        #             FlexibleMarginCallback(
        #                 initial_margin=args.initial_margin,
        #                 final_margin=args.final_margin,
        #                 total_epochs=args.tune_no
        #             ),
        #             MLflowLoggerCallback()
        #         ]
        #     )
            
        # Define Loss Function (Pass Model Here)
        loss_func = tl.WrappedTripletLoss(model=triplet_model, margin=args.margin)

        # Initialize Learner
        learner = Learner(dls, triplet_model, loss_func=loss_func, metrics=[], cbs=[]).to_fp16()

        # Add Callbacks
        learner.add_cb(LoggingCallback())
        learner.add_cb(ConstantMarginCallback())

        # Train the model
        print("Training the model.")
        learner.fine_tune(args.tune_no, freeze_epochs=0, base_lr=0.001)

        # Save the trained model
        torch.cuda.empty_cache()
        mh.save_model(triplet_model, model_folder_path)
        print(f"Process finished for Fold {fold+1}/{n_splits}")

        # Visualize a few batches (e.g., from the training loader) and save them in the model folder
        hf.save_and_visualize_batches(train_loader, model_folder_path, num_batches=10)

        # Plot the training and validation loss over time
        learner.recorder.plot_loss()
        # Save the current figure to your model folder (model_folder_path should be defined)
        plt.savefig(os.path.join(model_folder_path, "loss_plot.png"))
        plt.close()  # Close the figure to free memory

    print("\n Trining Completed")
