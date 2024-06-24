import os
import timm
import fastai
from fastai.vision.all import *
import torchvision
import torch.backends.cudnn as cudnn
import paths as p
import helper_functions as hf
import model_helpers as mh
from helper_functions import create_model

# Make cuda report the error where it actually occurs
CUDA_LAUNCH_BLOCKING=1

def data_checks():
    """ Performs specific data checks for adaptation dataset"""
    # Check if number of folders matches the labels
    print("This is sample data.")
    print("This is full dataset.")
    print("The number of folders does not match number of classes in sample data.")
    print("The number of folders does not match number of classes in the full dataset.")
    pass

def create_folder_structure(args):
    main_path = args.model
    os.makedirs(main_path, exist_ok=True)

    subfolder_name = "_".join([f"{k}_{v}" for k, v in vars(args).items() if v is not None])
    subfolder_path = os.path.join(main_path, subfolder_name)
    os.makedirs(subfolder_path, exist_ok=True)

    return subfolder_path


def main(args):
    model_folder_path = create_folder_structure(args)
    # Further processing in main
    print(f"Main function processing with folder path: {model_folder_path}")



if __name__ == '__main__':

    args = hf.parse_args()
    model_folder_path = create_folder_structure(args)
    main(args)

    # Set n_out based on the value of data_mode
    if args.data_mode == 'sample':
        n_out = 5
    elif args.data_mode == 'full_data':
        n_out = 30
    else:
        raise ValueError("Invalid data_mode. Choose 'sample' or 'full_data'.")

    print("Loading data.")
    dls = hf.load_data(args.data_mode, args.batch_size)

    #print("Visualizing original images.")
    # dls.show_batch()
    # plt.savefig('original_images.png')  # Save the displayed batch of images as a PNG file
    print("Data agumenting.")
    dls_aug = hf.augument_data(dls)
    #print("Visualizing agumented images.")
    # dls.show_batch()
    # plt.savefig('agumented_images.png')  # Save the displayed batch of images as a PNG file

    print("Create body.")
    body = create_body(hf.create_model(args.model))
    nf = num_features_model(body)
    print("Number of features: " + str(nf))
    head = create_head(nf, dls_aug.c, concat_pool=True)
    # Wrap into sequential to train the model
    net = nn.Sequential(body, head)
    print("Training model.")
    trained_model = mh.train_model(dls=dls_aug, model=args.model, n_out= n_out, tune_no = args.tune_no, lr=args.learning_rate)
    #get_model_params(trained_model)
    save_model(trained_model, model_folder_path)
    print("Process finished.")