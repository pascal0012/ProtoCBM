import h5py
import numpy as np
import torch

from utils_protocbm.train_utils import accuracy

#['activation_labels', 'activations', 'class_labels', 'preds']
file_path1 = "/home/ms66gide/outputs/Inspection/Waterbirds/waterbirds_visualization_attention/attribute_and_predictions_waterbirds.h5"
file_path2 = "/home/ms66gide/outputs/CBMWBIndependent/waterbirds_visualization_cam/attribute_and_predictions_waterbirds.h5"

avg_activation_diff = []
avg_activation_diff_when_pred_diff = []
average_pred_diff = []
accuracy1 = []
accuracy2 = []

batch_sum1 = 0
batch_sum2 = 0

diff_batch_sum = 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

with h5py.File(file_path1, "r") as f1, h5py.File(file_path2, "r") as f2:
    print(len(f1), len(f2))
    
    n_epochs = len(f1)

    for epoch_idx in range(n_epochs):

        # Access the dataset for the current epoch
        dataset1 = f1[f"epoch_{epoch_idx}"]
        dataset2 = f2[f"epoch_{epoch_idx}"]

        assert dataset1["class_labels"][:].shape == dataset2["class_labels"][:].shape, "Class label shapes differ between datasets"
        assert np.array_equal(dataset1["class_labels"][:], dataset2["class_labels"][:]), "Class labels differ between datasets"

        class_preds1 = dataset1["preds"][:]
        class_preds2 = dataset2["preds"][:]
        
        class_preds1 = np.argmax(class_preds1, axis=1)
        class_preds2 = np.argmax(class_preds2, axis=1)

        #mask where preds differ
        pred_diff_mask = class_preds1 != class_preds2

        #sanity check collect acc        
        accuracy1.append(len(dataset1["preds"][:]) * accuracy(torch.tensor(dataset1["preds"][:], device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')), torch.tensor(dataset1["class_labels"][:], device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')))[0].cpu())
        accuracy2.append(len(dataset2["preds"][:]) * accuracy(torch.tensor(dataset2["preds"][:], device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')), torch.tensor(dataset2["class_labels"][:], device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')))[0].cpu())

        batch_sum1 += len(dataset1["preds"][:])
        batch_sum2 += len(dataset2["preds"][:])

        # Read the activation labels
        activations1 = dataset1["activations"][:]
        activations2 = dataset2["activations"][:]

        #make prediction (sigmoid + threshold)

        activation_preds1 = (sigmoid(activations1) > 0.5).astype(float)
        activation_preds2 = (sigmoid(activations2) > 0.5).astype(float)
        
        #calculate average activation pred difference
        #sum diff per sample
        activation_diff = np.sum(np.abs(activation_preds1 - activation_preds2), axis=1)
        avg_activation_diff.append(len(activation_diff) * np.mean(activation_diff))

        #do the same for only samples where class preds differ
        activation_diff_when_pred_diff = activation_diff[pred_diff_mask]
        avg_activation_diff_when_pred_diff.append(len(activation_diff_when_pred_diff) * np.mean(activation_diff_when_pred_diff))

        diff_batch_sum += len(activation_diff_when_pred_diff)

        average_pred_diff.append(len(pred_diff_mask) * np.mean(pred_diff_mask))

#average the collectors
final_avg_activation_diff = np.sum(avg_activation_diff) / batch_sum1
final_avg_activation_diff_when_pred_diff = np.sum(avg_activation_diff_when_pred_diff) / diff_batch_sum
final_average_pred_diff = np.sum(average_pred_diff) / batch_sum1

print(batch_sum1, batch_sum2, diff_batch_sum)
accuracy1 = np.sum(accuracy1) / batch_sum1
accuracy2 = np.sum(accuracy2) / batch_sum2
#accuracy1 = np.mean(accuracy1)
#accuracy2 = np.mean(accuracy2)

print(f"Accuracy of model 1: {accuracy1}")
print(f"Accuracy of model 2: {accuracy2}")

print(f"Average activation prediction difference across epochs: {final_avg_activation_diff}")
print(f"Average activation prediction difference when class predictions differ: {final_avg_activation_diff_when_pred_diff}")
print(f"Average class prediction difference across epochs: {final_average_pred_diff}")

