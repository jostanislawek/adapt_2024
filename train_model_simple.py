import os
import timm
import fastai
from fastai.vision.all import *
import torchvision
import torch.backends.cudnn as cudnn
import paths as p
import helper_functions as hf

# Make cuda report the error where it actually occurs
CUDA_LAUNCH_BLOCKING=1

def load_data():
    # dls = ImageDataLoaders.from_folder(p.data_full, train="Training", valid="Validation",
    #                                    item_tfms=Resize(224), bs=4, num_workers=0)
    dls = ImageDataLoaders.from_folder(p.data_full, train="Training", valid="Validation",
                                       item_tfms=Resize(224), bs=4)
    return dls


def train_model(dls):
    model = vision_learner(dls, 'convnext_tiny', normalize=True, n_out=30, opt_func=Adam)
    model.fine_tune(2)


def save_model(model):
    torch.save(model, p.model_path + 'convnext_tiny.pt')
    torch.save(model.state_dict(), p.model_path + 'convnext_tiny.pt')


if __name__ == '__main__':
    print("Loading data.")
    dls = load_data()
    print("Create body.")
    body = create_body(convnext_tiny())
    nf = num_features_model(body)
    print("Number of features: " + str(nf))
    head = create_head(nf, dls.c, concat_pool=True)
    # Wrap into sequential to train the model
    net = nn.Sequential(body, head)
    print("Training model.")
    trained_model = train_model(dls)
    save_model(trained_model)
    print("Process finished.")