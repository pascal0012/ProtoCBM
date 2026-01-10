"""
Evaluate trained models on the SUB (Synthetic Attribute Substitutions) benchmark dataset.
This evaluates how well concept-based models generalize to novel attribute combinations.
"""

import os

import torch
from tqdm import tqdm

from cub.dataset import SUBDataset
from utils_protocbm.train_utils import (
    accuracy,
    gather_args,
    prepare_model,
    create_model,
    AverageMeter,
)
from utils_protocbm.eval_utils import get_eval_transform_for_model
from torch.utils.data import DataLoader
from cub.config import BASE_DIR


def eval_sub(args):
    """
    Evaluate model on SUB dataset.
    Returns class accuracy (bird species prediction).
    """

    # Create the model and load weights
    model = create_model(args)
    model, device = prepare_model(model, args, load_weights=True)
    model.eval()

    # Get transforms for the model
    transform = get_eval_transform_for_model(model, args)[0]

    # Load SUB dataset
    sub_data_dir = getattr(args, "sub_data_dir", os.path.join(BASE_DIR, "data/SUB"))
    if not os.path.isabs(sub_data_dir):
        sub_data_dir = os.path.join(BASE_DIR, sub_data_dir)
    sub_limit = getattr(args, "sub_limit", None)

    print(f"Loading SUB dataset from {sub_data_dir}")
    dataset = SUBDataset(sub_data_dir, transform=transform, limit=sub_limit)

    # Get valid attribute names (filtered if only_cub_attributes=True)
    valid_attr_names = dataset.get_valid_attr_names()
    num_valid_attrs = dataset.num_valid_attributes()

    print(f"Dataset size: {len(dataset)} samples")
    print(f"Bird classes: {len(dataset.bird_names)}")
    print(f"Attribute classes: {num_valid_attrs}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )

    class_acc_meter = AverageMeter()

    # Track per-bird and per-attribute accuracies (using filtered attr count)
    bird_correct = torch.zeros(len(dataset.bird_names))
    bird_total = torch.zeros(len(dataset.bird_names))
    attr_correct = torch.zeros(num_valid_attrs)
    attr_total = torch.zeros(num_valid_attrs)

    with torch.no_grad():
        for data_idx, data in enumerate(tqdm(loader, desc="Evaluating SUB")):
            inputs, bird_labels, attr_labels = data
            inputs = inputs.to(device)
            bird_labels = bird_labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Get class predictions (bird species)
            if isinstance(outputs, (list, tuple)):
                class_outputs = outputs[0]  # First output is class prediction
            else:
                class_outputs = outputs

            # Calculate accuracy
            class_acc = accuracy(class_outputs, bird_labels, topk=(1,))
            class_acc_meter.update(class_acc[0].item(), inputs.size(0))

            # Track per-bird accuracy
            _, preds = class_outputs.max(1)
            for i in range(inputs.size(0)):
                bird_idx = bird_labels[i].item()
                orig_attr_idx = attr_labels[i].item()
                # Convert to filtered index for tracking
                filtered_attr_idx = dataset.get_filtered_attr_index(orig_attr_idx)

                # count number of occurance / normalization
                bird_total[bird_idx] += 1
                attr_total[filtered_attr_idx] += 1

                # correct prediction
                if preds[i] == bird_labels[i]:
                    bird_correct[bird_idx] += 1
                    attr_correct[filtered_attr_idx] += 1

    # Calculate per-class accuracies
    bird_acc = bird_correct / (bird_total + 1e-10)
    attr_acc = attr_correct / (attr_total + 1e-10)

    return class_acc_meter, bird_acc, attr_acc, dataset, valid_attr_names


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args = gather_args()

    # Ensure dataset is set to SUB
    if not hasattr(args, "dataset") or args.dataset != "sub":
        print(
            "Note: Running SUB evaluation (set dataset: sub in config for explicit selection)"
        )

    print("\n" + "=" * 60)
    print("SUB Dataset Evaluation")
    print("=" * 60 + "\n")

    # Run evaluation
    class_acc_meter, bird_acc, attr_acc, dataset, valid_attr_names = eval_sub(args)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nOverall Bird Classification Accuracy: {class_acc_meter.avg:.2f}%")

    print(f"\nMean Per-Bird Accuracy: {bird_acc.mean().item() * 100:.2f}%")
    print(f"Mean Per-Attribute Accuracy: {attr_acc.mean().item() * 100:.2f}%")

    print("\n--- Per-Bird Accuracies ---")
    for i, name in enumerate(dataset.bird_names):
        print(f"  {name}: {bird_acc[i].item() * 100:.2f}%")

    print("\n--- Per-Attribute Accuracies ---")
    print("(Correctly predicting the changed attribute)")
    for i, name in enumerate(valid_attr_names):
        print(f"  {name}: {attr_acc[i].item() * 100:.2f}%")
