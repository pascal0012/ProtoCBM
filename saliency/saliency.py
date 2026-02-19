import numpy as np
import torch
import torch.nn as nn
from captum.attr import LayerAttribution, LayerGradCam

from saliency.utils import find_fist_conv


def get_saliency_map_and_scores_and_prediction(model, inputs, args, attr_labels=None):
    """
    Given a saliency method, a salieny method for the current model is created and returned. Alongside it,
    the model forward pass will be perfomed and the prediction will be extracted from the output tuple and
    returned. Also, a similarity score will be returned per attribute.
    If the model does NOT support this, it'll simply return a dummy value.

    Args:
        model: The model
        inputs: The input to the model (image)
        args: The model arguments to identify the used model and desired saliency method
        attr_labels: The attribute gt values, needed for APN
    Returns:
        pred: The final class predictions [B, C]
        scores: The per-attribute scores [B, A]
        map: The saliency map of the provided method [B, A, H, W]
    """

    if args.saliency_method == "attention":
        assert attr_labels is not None, "APN forward pass requires attribute labels"
        preds, similarity_scores, attention_maps = model(inputs, attr_labels)
        
        if args.concept_mapper != "protomod":
            raise ValueError(
                f"Saliency method attention was selected, but concept_mapper is {args.concept_mapper}, must be protomod!"
            )
        new_maps = get_protomod_attention(attention_maps)
        return preds, similarity_scores, new_maps

    elif args.saliency_method == "cam":

        if args.model_name == "protocbm":
            from saliency.wrapper import WrapperProtoCBM
            preds, similarity_scores, attention_maps = model(inputs, attr_labels)

            wrapped_model = WrapperProtoCBM(model, attr_labels=attr_labels)
            preds, similarity_scores, attention_maps = model(inputs, attr_labels)

            wrapped_model = WrapperProtoCBM(model, attr_labels=attr_labels)

            attribute_maps = torch.ones((inputs.shape[0], args.n_attributes, 8, 8))
            for target in range(args.n_attributes):
                current_cam = calculate_cam(wrapped_model, inputs, target=target)
                attribute_maps[:, target] = current_cam[:, 0, :, :]

            return preds, similarity_scores, attribute_maps

        elif args.model_name == "cbm":
            from saliency.wrapper import WrapperCUB
            out = model(inputs, attr_labels)
            class_pred, attributes = out[0], out[1:]
            attributes = torch.stack(attributes, dim=1).detach().squeeze(-1)

            # iterate over the attributes and calculate CAMs
            attribute_maps = torch.ones((inputs.shape[0], args.n_attributes, 8, 8))
            for target in range(args.n_attributes):
                wrapped_model = WrapperCUB(model, out_index=target)
                current_cam = calculate_cam(wrapped_model, inputs, attr_labels, target=0)
                attribute_maps[:, target] = current_cam[:, 0, :, :]

            return class_pred, attributes, attribute_maps

        else:
            raise ValueError(
                f"CAM saliency method is only supported for protocbm model, got {args.model_name}!"
            )
        
        

    raise ValueError(f"Got invalid saliency method {args.saliency_method}")


def get_protomod_attention(attention_maps):
    # For a protomod model, the output will be (predictions, similarity_scores, attention_maps)
    # Normalize attention maps into [0, 1] range
    att_min = attention_maps.amin(dim=(2, 3), keepdim=True)
    att_max = attention_maps.amax(dim=(2, 3), keepdim=True)
    return (attention_maps - att_min) / (att_max - att_min + 1e-7)


def calculate_cam(
    wrapped_model: nn.Module,
    input_im: torch.Tensor,
    attr_labels: torch.Tensor,
    target: list[int],
    source_conv=None,
) -> torch.Tensor:
    """Calculate Class Activation Map for a given input and target class."""

    # get the last conv layer
    if source_conv is None:
        # find the last convolutional layer automatically
        last_conv = find_fist_conv(wrapped_model)

    layer_gc = LayerGradCam(wrapped_model, last_conv)
    attr = layer_gc.attribute((input_im, attr_labels), target)

    return attr
