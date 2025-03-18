import os
import torch
from fastai.vision.all import *
import helper_functions as hf
import model_helpers as mh
import triplet_loss as tl
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
from triplet_loss import TripletModel

def knn_eval(test_embeddings, test_labels, train_embeddings, train_labels):
    """
    Evaluate k-nearest neighbor accuracy (overall and per-class) for k = 1, 2, 3.

    Parameters:
        test_embeddings (np.ndarray): Embeddings for the test set.
        test_labels (np.ndarray): True labels for the test set.
        train_embeddings (np.ndarray): Embeddings for the training set.
        train_labels (np.ndarray): True labels for the training set.
    """
    # Define the k values you want to evaluate
    k_values = [1, 2, 3]

    # Dictionaries to store the number of correct predictions overall for each k
    overall_correct = {k: 0 for k in k_values}

    # Dictionaries to store per-class correct predictions and total samples
    per_class_correct = {k: defaultdict(int) for k in k_values}
    per_class_total = {k: defaultdict(int) for k in k_values}

    # Loop over each test sample
    for i in range(len(test_embeddings)):
        # Compute cosine similarities between the i-th test embedding and all training embeddings
        similarities = cosine_similarity(test_embeddings[i].reshape(1, -1), train_embeddings).flatten()
        # Sort indices by descending similarity (largest similarity first)
        sorted_indices = np.argsort(similarities)[::-1]
        
        # True label of the test sample
        true_label = test_labels[i]
        
        # Evaluate for each k value
        for k in k_values:
            # Get the indices of the top k nearest neighbors
            knn_indices = sorted_indices[:k]
            # Retrieve their corresponding labels
            knn_labels = [train_labels[idx] for idx in knn_indices]
            # Use majority vote to decide the predicted label (ties return the first most common)
            predicted_label = Counter(knn_labels).most_common(1)[0][0]
            
            # Update total count for this true label
            per_class_total[k][true_label] += 1
            
            # Check if the prediction is correct
            if predicted_label == true_label:
                overall_correct[k] += 1
                per_class_correct[k][true_label] += 1

    # Calculate overall accuracy for each k
    overall_accuracy = {k: overall_correct[k] / len(test_labels) for k in k_values}

    # Calculate per-class accuracy for each k
    per_class_accuracy = {k: {} for k in k_values}
    for k in k_values:
        for cls, total in per_class_total[k].items():
            per_class_accuracy[k][cls] = per_class_correct[k][cls] / total

    # Print the results
    for k in k_values:
        print(f"KNN (k={k}) Overall Accuracy: {overall_accuracy[k]:.4f}")
        print("Per Class Accuracy:")
        for cls, acc in per_class_accuracy[k].items():
            print(f"  Class {cls}: {acc:.4f}")
        print()  # For better readability between different k values

if __name__ == "__main__":
    hf.check_gpu()
    args = hf.parse_args()
    print(args)

    # Model path
    MODEL_PATH = "/mnt/d/PhD/Models/Adaptation_2024/convnext_tiny/Triplet_Loss/model_convnext_tiny_data_mode_full_data_batch_size_4_tune_no_10_resume_False/"

    # Ensure model path exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model directory not found: {MODEL_PATH}")

    print(f"Testing Model from Path: {MODEL_PATH}")

    # Load test data
    print("Loading Test Data")
    dls = hf.load_data(args.data_mode)  # Load dataset
    test_dl = dls.valid  # Extract only the test set

    # Load the triplet model
    triplet_model_path = os.path.join(MODEL_PATH, "model.pt")

    if os.path.exists(triplet_model_path):
        print("\nEvaluating Triplet Model on Test Data")

        # Load and prepare the model
        triplet_model = torch.load(triplet_model_path)
        triplet_model = triplet_model.to("cuda")
        triplet_model.eval()

        # Extract embeddings for test images
        test_embeddings, test_labels = [], []
        with torch.no_grad():
            for x, y in test_dl:
                x = x.to("cuda")
                emb = triplet_model.get_embedding(x)
                emb = emb / emb.norm(p=2, dim=1, keepdim=True)  # Normalize embeddings
                test_embeddings.append(emb.cpu().numpy())
                test_labels.append(y.cpu().numpy())

        test_embeddings = np.vstack(test_embeddings)
        test_labels = np.concatenate(test_labels)

        # Extract embeddings for training images (to compare against)
        train_embeddings, train_labels = [], []
        with torch.no_grad():
            for x, y in dls.train:
                x = x.to("cuda")
                emb = triplet_model.get_embedding(x)
                emb = emb / emb.norm(p=2, dim=1, keepdim=True)  # Normalize embeddings
                train_embeddings.append(emb.cpu().numpy())
                train_labels.append(y.cpu().numpy())

        train_embeddings = np.vstack(train_embeddings)
        train_labels = np.concatenate(train_labels)

        # Run KNN evaluation
        knn_eval(test_embeddings, test_labels, train_embeddings, train_labels)

        # Extract embeddings from the test dataloader
        test_embeddings, test_labels = [], []
        with torch.no_grad():
            for x, y in test_dl:
                x = x.to("cuda")
                emb = triplet_model.get_embedding(x)
                test_embeddings.append(emb.cpu().numpy())
                test_labels.append(y.cpu().numpy())

            test_embeddings = np.vstack(test_embeddings)
            test_labels = np.concatenate(test_labels)

            # Now visualize using TSNE and save the plot
            tsne_save_path = os.path.join(MODEL_PATH, "tsne_embeddings.png")
            hf.visualize_embeddings_tsne(test_embeddings, test_labels, tsne_save_path)
            print(f"t-SNE plot saved to {tsne_save_path}")

    else:
        print("Triplet Model not found, skipping evaluation.")