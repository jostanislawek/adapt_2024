import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
from fastai.vision.all import *
import helper_functions as hf
import model_helpers as mh
from sklearn.model_selection import StratifiedKFold
import numpy as np


"""
HOW TO TRAIN AND RESUME TRAINING
- python train_model.py -m "convnext_tiny" -d "sample" -bs 4 -tn 1 --note "initial train"

HOW TO CONTINUE TRAINING SAME MODEL WITH THE SAME NAME, BATCH SIZE AND DATA MODE:
- python train_model.py -m "convnext_tiny" -d "sample" -bs 4 -tn 1 --note "continuing train" --resume
"""


def main(args, fold_index):
    """ Main function for training and saving models across CV folds. """
    model_folder_path = hf.create_folder_structure(args, fold_index)
    if not model_folder_path:
        raise ValueError("Failed to create model folder path.")
    print(f"Main function processing with folder path: {model_folder_path}")
    return model_folder_path 


if __name__ == '__main__':
    # Verify that the environment variables are set
    hf.check_gpu()
    args = hf.parse_args()
    print(args)

    n_splits = 5  # Use 5-Fold Stratified CV

    for fold in range(n_splits):
        print(f"\n Training Fold {fold+1}/{n_splits}")

        # Load stratified cross-validation split
        data = hf.load_data_crossval_stratified(args, args.data_mode, n_splits=n_splits, fold_index=fold)

        train_dls = data["train_dls"]
        valid_dls = data["valid_dls"]

        # Create model folder per fold
        model_folder_path = hf.create_folder_structure(args, fold)

        # Load model architecture
        model_arch = getattr(models, args.model, None)
        if model_arch is None:
            raise ValueError(f"Invalid model architecture: {args.model}")

        # Set n_out based on data_mode
        if args.data_mode == 'sample':
            n_out = 5
        elif args.data_mode == 'full_data':
            n_out = 30
        else:
            raise ValueError("Invalid data_mode. Choose 'sample' or 'full_data'.")

        print("Data augmenting.")
        dls_aug = hf.augment_data(train_dls, n_out)

        print("Create body.")
        body = create_body(mh.create_model_object(args.model))
        nf = num_features_model(body)
        print(f"Number of features: {nf}")
        head = create_head(nf, dls_aug.c, concat_pool=True)

        # Wrap into sequential model
        net = nn.Sequential(body, head)
        print("Training model.")

        trained_model = mh.train_model(
            dls=dls_aug,
            model=args.model,
            n_out=n_out,
            tune_no=args.tune_no,
            lr=args.learning_rate if args.learning_rate is not None else 0.01,
            model_path=f"model_fold{fold}.pth",
            resume=args.resume
        )

        torch.cuda.empty_cache()

        # Save the trained model
        mh.save_model(trained_model, model_folder_path)
        print(f"Process finished for Fold {fold+1}/{n_splits}")

    print("\n Cross-Validation Completed for All Folds")
