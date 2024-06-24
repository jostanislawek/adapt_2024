"""This module contains all the functions which are used in multiple notebooks"""
import numpy as np
import matplotlib.pyplot as plt
from fastai.vision.all import *
from fastai.vision.models import resnet18, resnet34, resnet50, resnet101, resnet152, convnext_tiny, convnext_small
import argparse
import paths as p

def check_gpu():
    """ Check the enviroment and torch version """

    print(torch.__version__)
    print("cuda:0" if torch.cuda.is_available() else "cpu")

def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def predict_image(img):

  img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()])

  load_image = Image.open(img)
  #load_image = img_transforms(img).to(device)
  transform_image = img_transforms(load_image)
  image_transformed = torch.unsqueeze(transform_image, 0)

  # Set model to eval
  model.eval()

  # Get prediction
  prediction = F.softmax(model(image_transformed), dim=1)
  prediction = prediction.argmax()
  prediction

  labels = ['AcornSquash', 'Avocado', 'Berry', 'CustardApple', 'Raspberry']
  print(labels[prediction])


def create_model(model_name):
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


def load_data(data_type, batch_size):
    """ Loads data for training or finetuning.
    Args:
    data_type: str
    Takes arguments "sample" or "full_data".
    batch_size: int
    Variable to asjust batch size.

    Returns:
    dls: data loader object
    Return data in data loader object. """

    if data_type == 'sample':
        dls = ImageDataLoaders.from_folder(p.data_train_sample, train="Train", valid="Validation",
                                           item_tfms=Resize(224), bs=4, num_workers=0)
    elif data_type == 'full_data':
        dls = ImageDataLoaders.from_folder(p.data_full, train="Train", valid="Validation",
                                           item_tfms=Resize(224), bs=24, num_workers=2)
    else:
        raise ValueError("Invalid data type. Choose 'sample' or 'full_data'.")
    return dls

def augument_data(data):
    "Adding item transformations"
    data.new(
        item_tfms=RandomResizedCrop(224, min_scale=0.5, max_scale=0.25),
        tflip=DihedralItem(p=0.5),
        batch_tfms=aug_transforms(mult=2),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    )
    return data
