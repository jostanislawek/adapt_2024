import os
import torch
from fastai.vision.all import *
import helper_functions as hf
import model_helpers as mh
import triplet_loss as tl
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

hf.check_gpu()
args = hf.parse_args()
print(args)

# Paste the model path here (Change if needed)
MODEL_PATH = "/mnt/d/PhD/Models/Adaptation_2024/convnext_tiny/Baseline_Softmax/model_convnext_tiny_data_mode_full_data_batch_size_24_tune_no_25_resume_False/"

# Ensure model path exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model directory not found: {MODEL_PATH}")

print(f"Testing Model from Path: {MODEL_PATH}")

# Load test data
print("\Loading Test Data")
dls = hf.load_data(args.data_mode)  # Load dataset
test_dl = dls.valid  # Extract only the test set

# Load Softmax Model (PyTorch)
softmax_model_path = os.path.join(MODEL_PATH, "model.pt")
if os.path.exists(softmax_model_path):
    print("Evaluating Softmax Model on Test Data")
    
    # Load the PyTorch model
    softmax_model = torch.load(softmax_model_path)
    softmax_model.eval()

    # Compute accuracy manually
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_dl:
            images, labels = images.to("cuda"), labels.to("cuda")  # Move to GPU if available
            outputs = softmax_model(images)
            _, predicted = torch.max(outputs, 1)  # Get class prediction
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    softmax_test_accuracy = correct / total
    print(f"Softmax Model Test Accuracy: {softmax_test_accuracy:.4f}")

else:
    print(" Softmax Model not found, skipping evaluation.")