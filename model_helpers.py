import os

import torch
from fastai.callback.tracker import SaveModelCallback
from fastai.vision.all import *
from fastai.vision.learner import _update_first_layer
from fastai.vision.models import (convnext_small, convnext_tiny, resnet18,
                                  resnet34, resnet50, resnet101, resnet152)
from torchvision.models import ConvNeXt_Tiny_Weights

import paths as p
import triplet_loss as tl


def create_model_object(model_name):
    if model_name == 'resnet18':
        return resnet18()
    elif model_name == 'resnet34':
        return resnet34()
    elif model_name == 'resnet50':
        return resnet50()
    elif model_name == 'resnet101':
        return resnet101()
    elif model_name == 'resnet152':
        return resnet152()
    elif model_name == 'convnext_tiny':
        return convnext_tiny()
    elif model_name == 'convnext_small':
        return convnext_small()
    else:
        raise ValueError("Invalid model name.")


def train_model(dls, model, n_out, tune_no, lr=0.1, loss_func=None, metrics=[accuracy], cbs=None, model_path="model.pth", resume=False):

    """
    Train a model with optional checkpoint loading for resuming training.

    Args:
        dls: DataLoaders object.
        model: Model architecture.
        n_out: Number of output classes.
        tune_no: Number of fine-tuning epochs.
        lr: Learning rate.
        loss_func: Loss function.
        opt_func: Optimizer function.
        metrics: Training metrics.
        cbs: List of callbacks.
        model_path: Path to save/load the model checkpoint.
        resume: Whether to resume training from a saved checkpoint.

    Returns:
        Trained model.
    """
    if cbs is None:
        cbs = []
    
    learn = vision_learner(dls, model, normalize=True, n_out=n_out, loss_func=loss_func, cbs=cbs, metrics=metrics)

    # Remove .pth from model_path for fastai
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    # Check the data loaders
    for batch in dls.train:
        x, y = batch
        # Ensure x and y have expected shapes
        assert x.ndim in {4}, f"Unexpected x shape: {x.shape}"
        assert y.ndim == 1, f"Unexpected y shape: {y.shape}"
        assert x.size(0) == y.size(0), f"Batch size mismatch: x.size(0)={x.size(0)}, y.size(0)={y.size(0)}"
        print(f"Batch x shape: {x.shape}, Batch y shape: {y.shape}")

    # Load previous checkpoint if resuming training
    if resume and os.path.exists(model_path):
        print(f"Resuming training from: {model_path}")
        learn.load(model_path.replace(".pth", ""))  # Remove .pth for FastAI loading
    else:
        print("Starting fresh training.")

    # Include SaveModelCallback to save best model
    # learn.fine_tune(tune_no, base_lr=slice(lr/10, lr), 
    #             cbs=[ShowGraphCallback(), SaveModelCallback(fname=model_name, with_opt=True)] + cbs)
    
    if lr is None:
        raise ValueError("Learning rate (lr) cannot be None. Please specify a valid value.")

    #If don't want progresive LR
    learn.fine_tune(tune_no, base_lr=lr, 
                cbs=[ShowGraphCallback(), SaveModelCallback(fname=model_name, with_opt=True)] + cbs)

    #learn.fine_tune(tune_no, lr_max=lr, cbs=[ShowGraphCallback(), SaveModelCallback(fname=model_name, with_opt=True)] + cbs)

    # Save final model
    learn.export(model_path.replace(".pth", ".pkl"))  # Save for inference
    torch.save(learn.model.state_dict(), model_path)  # Save weights
    print(f"Model saved at {model_path}")

    return learn.model

def get_model_params(trained_model):
    # Accessing model parameters
    for name, param in trained_model.model.named_parameters():
        print(f"Parameter: {name}, Value: {param}")

def evaluate_model(model):
    """ Prints the model architecture """
    print(model.eval())
    return model

def save_model(model, path):
    model_path = os.path.join(path, "model.pt")
    torch.save(model, model_path)
