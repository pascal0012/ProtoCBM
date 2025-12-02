import numpy as np

import torch
import torch.nn as nn

from captum.attr import LayerGradCam, LayerAttribution

from saliency.utils import find_fist_conv


def get_saliency_map_and_scores_and_prediction(output, args):
    """
        Given a saliency method, a salieny method for the current model output is created and returned. Alongside it,
        the prediction will be extracted from the output tuple and returned. Also, a similarity score will be returned
        per attribute. If the model does NOT support this, it'll simply return a dummy value.

        Args:
            output: The output of any given model
            args: The model arguments to identify the used model and desired saliency method
        Returns:
            pred: The final class predictions [B, C]
            scores: The per-attribute scores [B, A]
            map: The saliency map of the provided method
    """
    if args.saliency_method == "attention":
        if args.concept_mapper != "protomod":
            raise ValueError(f"Saliency method attention was selected, but concept_mapper is {args.concept_mapper}, must be protomod!")
        return get_protomod_attention_and_predition(output)
    elif args.saliency_method == "cam":
        return None # TODO
    raise ValueError(f"Got invalid saliency method {args.saliency_method}")


def get_protomod_attention_and_predition(output):
    # For a protomod model, the output will be (predictions, similarity_scores, attention_maps)
    preds, similarity_scores, attention_maps = output

    # Normalize attention maps into [0, 1] range
    att_min = attention_maps.amin(dim=(2,3), keepdim=True)
    att_max = attention_maps.amax(dim=(2,3), keepdim=True)
    return preds, similarity_scores, (attention_maps - att_min) / (att_max - att_min + 1e-7)
