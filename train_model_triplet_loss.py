import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
from fastai.vision.all import *
import torch.backends.cudnn
import paths as p
import helper_functions as hf
import model_helpers as mh
import triplet_loss as tl
from torchvision import transforms


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
    
    #############################################################################
    ########################### LOAD AND AGUMENT DATA ###########################
    #############################################################################
    
    print("Loading data.")
    
    transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create datasets
    train_dataset = tl.TripletDataset(root_dir='/mnt/d/PhD/Images/Adaptation_Dataset_Sample/Train/', transform=transforms)
    valid_dataset = tl.TripletDataset(root_dir='/mnt/d/PhD/Images/Adaptation_Dataset_Sample/Validation/', transform=transforms)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    hf.save_and_visualize_batches(train_loader, '/mnt/d/PhD/Models/Adaptation_2024/batches/', 10)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=0)

    # Retrieve numerical labels directly from the TripletDataset and deduplicate them
    all_labels = train_dataset.labels  # Access the labels directly

    dls = DataLoaders(train_loader, valid_loader)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #############################################################################
    ########################## CREATE AND TRAIN MODEL ###########################
    #############################################################################

    print("Create body.")
    body = create_body(mh.create_model_object(args.model))
    nf = num_features_model(body)
    print("Number of features: " + str(nf))
    head = create_head(nf, n_out, concat_pool=True)

    # Wrap into sequential to train the model
    net = nn.Sequential(body, head)
    print("Training model.")
    model = mh.create_model_object(args.model)
    trained_model = mh.triplet_loss_learner(dls, model, n_out=n_out, tune_no = args.tune_no)
    #trained_model.fine_tune(tune_no = args.tune_no)

    torch.cuda.empty_cache()
    #get_model_params(trained_model)
    mh.save_model(trained_model, model_folder_path)
    print("Process finished.")