import os
import timm
import fastai
from fastai.vision.all import *
from fastai.callback.all import *
from timm import create_model
from fastai.vision.learner import _update_first_layer
import torchvision
import torch.backends.cudnn as cudnn
import paths as p
import triplet_loss as tl
from fastai.vision.models import resnet18, resnet34, resnet50, resnet101, resnet152, convnext_tiny, convnext_small


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


def train_model(dls, model, n_out, tune_no, lr=0.1, loss_func=None, opt_func=Adam, metrics="accuracy"):
    learn  = vision_learner(dls, model, normalize=True, n_out=n_out, loss_func=loss_func, opt_func=opt_func, lr=0.1)

    # Check the data loaders
    for batch in dls.train:
        x, y = batch
        # Ensure x and y have expected shapes
        assert x.ndim in {4}, f"Unexpected x shape: {x.shape}"
        assert y.ndim == 1, f"Unexpected y shape: {y.shape}"
        assert x.size(0) == y.size(0), f"Batch size mismatch: x.size(0)={x.size(0)}, y.size(0)={y.size(0)}"
        print(f"Batch x shape: {x.shape}, Batch y shape: {y.shape}")

    learn.fine_tune(tune_no, cbs=ShowGraphCallback())
    
    # Shows all the training steps and where callbacks are located
    print("SHOW TRAINING LOOP")
    model.show_training_loop()
    return learn

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


# def triplet_loss_learner(dls, model, n_out, tune_no, lr=0.1, opt_func=Adam, metrics="accuracy", cbs=None):

#     # Create the learner with the custom TripletLoss
#     learn = Learner(dls, model, loss_func=tl.TripletLoss(margin=1.0), 
#                     opt_func=opt_func, lr=lr, cbs=cbs, metrics=[metrics])

#     # Fine-tuning the model
#     learn.fine_tune(tune_no, cbs=[ShowGraphCallback()] + (cbs or []))
    
#     return learn

def triplet_loss_learner(dls, model, n_out, tune_no, lr=0.1, loss_func=None, opt_func=Adam, metrics="accuracy"):
    learn  = vision_learner(dls, model, normalize=True, n_out=n_out, loss_func=tl.TripletLoss(margin=1.0), opt_func=opt_func, lr=0.1)

    # Check the data loaders
    for batch in dls.train:
        x, y = batch
        # Ensure x and y have expected shapes
        assert x.ndim in {4}, f"Unexpected x shape: {x.shape}"
        assert y.ndim == 1, f"Unexpected y shape: {y.shape}"
        assert x.size(0) == y.size(0), f"Batch size mismatch: x.size(0)={x.size(0)}, y.size(0)={y.size(0)}"
        print(f"Batch x shape: {x.shape}, Batch y shape: {y.shape}")

    learn.fine_tune(tune_no, cbs=ShowGraphCallback())
    
    # Shows all the training steps and where callbacks are located
    print("SHOW TRAINING LOOP")
    model.show_training_loop()
    return learn