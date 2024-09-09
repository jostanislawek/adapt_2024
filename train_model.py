import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
import timm
import fastai
from fastai.vision.all import *
import torchvision
import torch.backends.cudnn as cudnn
import paths as p
import helper_functions as hf
import model_helpers as mh


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

    # Verify that the environment variables are set
    hf.check_gpu()

    args = hf.parse_args()
    print(args)
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
    dls = hf.load_data(args.data_mode)
    
    print("Data agumenting.")
    # Apply augmentations and check targets
    dls_aug = hf.augment_data(dls, n_out)

    print("Create body.")
    body = create_body(mh.create_model_object(args.model))
    nf = num_features_model(body)
    print("Number of features: " + str(nf))
    head = create_head(nf, dls_aug.c, concat_pool=True)

    # Wrap into sequential to train the model
    net = nn.Sequential(body, head)
    print("Training model.")
    trained_model = mh.train_model(dls=dls_aug, model=args.model, n_out=n_out, tune_no=args.tune_no, lr=args.learning_rate)
    torch.cuda.empty_cache()
    mh.save_model(trained_model, model_folder_path)
    print("Process finished.")