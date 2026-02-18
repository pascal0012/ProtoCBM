"""
Evaluate trained models on the official CUB test set
"""

import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from localization.part_seg_iou import compute_IoU_to_seg_masks, compute_mIoU_statistics
from localization.visualise import (
    visualise_part_segmentations, 
    visualise_localization_acc_boxes, 
    plot_threshold_curve,
    visualize_keypoint_distances
)
from localization.localization_accuracy import (
    compute_localization_distance, 
    calculate_average_partwise_localization_distance,
)
from saliency.saliency import get_saliency_map_and_scores_and_prediction
from utils_protocbm.mappings import MAP_RESULT_GROUPS_TO_CUB_GROUPS
from utils_protocbm.eval_utils import LocalizationMeter, get_localization_loader
from utils_protocbm.train_utils import AverageMeter, accuracy, binary_accuracy, prepare_model, create_model, gather_args


def eval(args):
    """
        TODO
    """

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Create the model and load weights
    model = create_model(args)
    model, device = prepare_model(model, args, load_weights=True)
    model.eval()

    # Get the localization data loader and additional transform statistics
    loader, transform_mean, transform_std, img_size = get_localization_loader(model, args.data_dir, args.split_dir, args)
    map_attr_id_to_part_seg_group = loader.dataset.map_attr_id_to_part_seg_group
    attribute_names = loader.dataset.attribute_names
    unmatched_attr_mask = loader.dataset.unmatched_attr_mask

    # Collecting metric values across batches for proper mean
    seg_loc_meter = LocalizationMeter(attribute_names, device)
    loc_acc_collector = []
    class_acc_meter = AverageMeter()
    attr_acc_meter = AverageMeter()
    attr_ce_meter = AverageMeter()

    # Collectors for confusion matrix
    all_preds = []
    all_labels = []
    all_attr_preds = []
    all_attr_labels = []
    all_attr_preds_raw = []  # without sigmoid, threshold at 0

    # Per-attribute TP/FP/FN counters for precision/recall/F1
    n_attrs = args.n_attributes
    attr_tp = torch.zeros(n_attrs, device=device)
    attr_fp = torch.zeros(n_attrs, device=device)
    attr_fn = torch.zeros(n_attrs, device=device)

    if args.dataset == "waterbirds":
        class_acc_meter_water = AverageMeter()
        attr_acc_meter_water = AverageMeter()
        class_acc_meter_land = AverageMeter()
        attr_acc_meter_land = AverageMeter()

    thresholds = np.arange(0, 1.05, 0.05)
    threshold_ious = torch.zeros((len(thresholds), len(attribute_names)), device=device)
    threshold_counts = torch.zeros((len(thresholds), len(attribute_names)), device=device)

    with torch.no_grad():
        for data_idx, data in enumerate(tqdm(loader, desc="Evaluating batches")):
            # Cast data to device
            data = [v.to(device) if torch.is_tensor(v) else v for v in data]

            if args.dataset == "waterbirds":
                inputs, labels, attr_labels, part_seg_masks, part_bbs, source_paths, part_gts, waterbirds_labels = data
            else:
                inputs, labels, attr_labels, part_seg_masks, part_bbs, source_paths, part_gts = data

            # attr_labels = torch.stack(attr_labels).t()  # N x A
            attr_labels = torch.stack(attr_labels, dim=1).float().to(device)

            # Pass through model, get model prediction and saliency map per attribute
            pred, scores, saliency_maps = get_saliency_map_and_scores_and_prediction(model, inputs, args, attr_labels=attr_labels)
            saliency_maps = saliency_maps.to(device)

            # Calculate classification accuracy
            class_acc = accuracy(pred, labels, topk=(1,))
            class_acc_meter.update(class_acc[0], pred.size(0))

            # Collect predictions and labels for confusion matrix
            all_preds.append(pred.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())

            # Calculate attribute accuracy
            attr_acc = binary_accuracy(scores, attr_labels)
            attr_acc_meter.update(attr_acc, pred.size(0))

            # Calculate attribute cross-entropy
            bce_loss = nn.BCEWithLogitsLoss()(scores, attr_labels.float())
            attr_ce_meter.update(bce_loss, pred.size(0))

            # Accumulate TP/FP/FN for precision/recall/F1
            attr_preds = (torch.sigmoid(scores) >= 0.5).float()
            attr_tp += (attr_preds * attr_labels).sum(dim=0)
            attr_fp += (attr_preds * (1 - attr_labels)).sum(dim=0)
            attr_fn += ((1 - attr_preds) * attr_labels).sum(dim=0)

            # Collect attribute predictions and labels for confusion matrix
            all_attr_preds.append(attr_preds.cpu())
            all_attr_labels.append(attr_labels.cpu())
            all_attr_preds_raw.append((scores >= 0).float().cpu())

            # For waterbirds: Calculate class accuracy and binary accuracy separated for water and landbirds
            if args.dataset == "waterbirds":

                # Extract water birds data
                water_mask = waterbirds_labels == 1
                pred_w   = pred[water_mask]
                labels_w = labels[water_mask]
                scores_w = scores[water_mask]
                attr_w   = attr_labels[water_mask]

                # Calculate accuracies for waterbirds
                if pred_w.size(0) > 0:
                    class_acc_w = accuracy(pred_w, labels_w, topk=(1,))
                    class_acc_meter_water.update(class_acc_w[0], pred_w.size(0))
                    attr_acc_w = binary_accuracy(scores_w, attr_w)
                    attr_acc_meter_water.update(attr_acc_w, pred_w.size(0))

                # Extract land birds data
                land_mask  = waterbirds_labels == 0
                pred_l   = pred[land_mask]
                labels_l = labels[land_mask]
                scores_l = scores[land_mask]
                attr_l   = attr_labels[land_mask]

                # Calculate accuracies for landbirds
                if pred_l.size(0) > 0:
                    class_acc_l = accuracy(pred_l, labels_l, topk=(1,))
                    class_acc_meter_land.update(class_acc_l[0], pred_l.size(0))
                    attr_acc_l = binary_accuracy(scores_l, attr_l)
                    attr_acc_meter_land.update(attr_acc_l, pred_l.size(0))
            
            # Compute localization accuracy and collect into our collector
            predicted_coords, dists, _, _ = compute_localization_distance(
                scores, 
                saliency_maps, 
                part_bbs, 
                part_gts,
                loader.dataset.part_dict,
                loader.dataset.map_part_to_attr_loc_acc, 
                loc_acc_collector, img_size=img_size,
                use_argmax=args.use_argmax
            )

            # Compute IoU between part segmentation masks and our saliency maps, for each attribute
            # Map out unmapped attributes from the saliency mask
            iou_scores, spr, cpr, saliency_maps_upsampled, seg_masks_per_attribute = compute_IoU_to_seg_masks(
                saliency_maps[:, unmatched_attr_mask], part_seg_masks, map_attr_id_to_part_seg_group, scores[:, unmatched_attr_mask]
            )
            seg_loc_meter.update(spr, cpr)

            # Compute curve statistics
            if args.plot_curve:
                for i, t in enumerate(thresholds):
                    _, tspr, tcpr, _, _ = compute_IoU_to_seg_masks(
                        saliency_maps[:, unmatched_attr_mask], part_seg_masks, map_attr_id_to_part_seg_group, scores, keep_threshold=t
                    )
                    threshold_ious[i] += tspr
                    threshold_counts[i] += tcpr

            # Visualise part segmentations with saliency
            if args.vis_every_n > 0 and data_idx % args.vis_every_n == 0:
                # Always visualize the first image in each batch for consistency
                batch_idx = 0

                visualise_part_segmentations(
                    inputs, saliency_maps_upsampled, seg_masks_per_attribute,
                    attribute_names, iou_scores,
                    batch_idx=batch_idx,
                    source_paths=source_paths, t_mean=transform_mean, t_std=transform_std, save_path=args.out_dir_part_seg
                )

                visualize_keypoint_distances(part_gts,
                                             inputs,
                                             source_paths,
                                             predicted_coords,
                                             dists,
                                             data_idx,
                                             list(loader.dataset.part_dict.values()),
                                             batch_idx=batch_idx,
                                             t_mean=transform_mean,
                                             t_std=transform_std,
                                             save_path=args.out_dir_part_seg
                )

    # Compute statistics over all batches
    seg_loc_meter.compute(map_attr_id_to_part_seg_group, verbose=True)

    if args.plot_curve:
        #stats for curve
        threshold_accs = [compute_mIoU_statistics(threshold_ious[i], threshold_counts[i], attribute_names, map_attr_id_to_part_seg_group, verbose=False) for i in range(len(thresholds))]
        plot_threshold_curve(thresholds, threshold_accs, args.out_dir_part_seg)


    # Compute part localization statistics, considering our grouping of attributes to groups.
    #calculate_average_partwise_localization_accuracy(loc_acc_collector, MAP_PART_SEG_GROUPS_TO_CUB_GROUPS, IoU_thr=args.IoU_threshold)
    calculate_average_partwise_localization_distance(loc_acc_collector, MAP_RESULT_GROUPS_TO_CUB_GROUPS)

    # Compute mean classification and attribute accuracies and print
    print("\n--------- ACCURACIES ---------\n")
    print(f"Mean Classification Accuracy: {class_acc_meter.avg.item():.4f}")
    print(f"Mean Attribute Accuracy: {attr_acc_meter.avg.item():.4f}")
    print(f"Mean Attribute Cross-Entropy: {attr_ce_meter.avg.item():.4f}")

    # Compute precision, recall, F1 from accumulated TP/FP/FN
    precision_per_attr = attr_tp / (attr_tp + attr_fp + 1e-8)
    recall_per_attr = attr_tp / (attr_tp + attr_fn + 1e-8)
    f1_per_attr = 2 * precision_per_attr * recall_per_attr / (precision_per_attr + recall_per_attr + 1e-8)

    macro_precision = precision_per_attr.mean().item()
    macro_recall = recall_per_attr.mean().item()
    macro_f1 = f1_per_attr.mean().item()

    print(f"\n--------- ATTRIBUTE METRICS (macro-avg over {n_attrs} attrs) ---------\n")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall:    {macro_recall:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}")

    # Classification report
    print("\n--------- CLASSIFICATION REPORT ---------\n")
    print(classification_report(all_labels, all_preds, zero_division=0))

    # Plot binary attribute confusion matrix (aggregated over all attributes)
    all_attr_preds = torch.cat(all_attr_preds).numpy().flatten()
    all_attr_labels = torch.cat(all_attr_labels).numpy().flatten()
    attr_cm = confusion_matrix(all_attr_labels, all_attr_preds, labels=[0, 1])
    tn, fp, fn, tp = attr_cm.ravel()

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=attr_cm,
        display_labels=["Absent (0)", "Present (1)"]
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Attribute Binary Confusion Matrix (all attributes)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    attr_cm_path = os.path.join(args.out_dir_part_seg, "attribute_confusion_matrix.png")
    plt.savefig(attr_cm_path, dpi=150)
    plt.close(fig)

    print("\n--------- ATTRIBUTE CONFUSION MATRIX (sigmoid >= 0.5) ---------")
    print(f"  TN={tn}  FP={fp}")
    print(f"  FN={fn}  TP={tp}")
    print(f"Attribute confusion matrix saved to {attr_cm_path}")

    # Plot binary attribute confusion matrix WITHOUT sigmoid (raw logits >= 0)
    all_attr_preds_raw = torch.cat(all_attr_preds_raw).numpy().flatten()
    attr_cm_raw = confusion_matrix(all_attr_labels, all_attr_preds_raw, labels=[0, 1])
    tn_r, fp_r, fn_r, tp_r = attr_cm_raw.ravel()

    # For waterbirds: Show accuracy metrics separated by land and water
    if args.dataset == "waterbirds":
        print(f"\nMean Waterbirds Classification Accuracy: {class_acc_meter_water.avg.item():.4f}")
        print(f"Mean Waterbirds Attribute Accuracy: {attr_acc_meter_water.avg.item():.4f}")

        print(f"\nMean Landbirds Classification Accuracy: {class_acc_meter_land.avg.item():.4f}")
        print(f"Mean Landbirds Attribute Accuracy: {attr_acc_meter_land.avg.item():.4f}")


if __name__ == '__main__':
    # Set deterministic behavior for reproducibility
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

    args = gather_args()

    # Create out folder for any visualizations / eval outputs
    out_folder_path = os.path.join(args.log_dir, f"{args.dataset}_visualization_{args.saliency_method}")
    os.makedirs(out_folder_path, exist_ok=True)
    args.out_dir_part_seg = out_folder_path

    # Print everything into separate file
    path_to_output_txt = os.path.join(args.out_dir_part_seg, "eval.txt")
    print(f"Writing outputs into {path_to_output_txt}.")
    sys.stdout = open(path_to_output_txt, 'a')

    # Print all args
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    # Run main evaluation
    eval(args)
