
import torch
import torch.nn as nn

class WrapperCUB(nn.Module):
    """
    Wrap a model that returns a list into one that returns a tensor.
    You can choose which element (or combine them) for attribution.
    """
    def __init__(self, model, out_index=0, attr_labels=None, is_independent=False):
        super().__init__()
        self.model = model
        self.out_index = out_index
        self.attr_labels = attr_labels
        self.is_independent = is_independent

    def forward(self, input_im, attr_labels):
        out = self.model(input_im, attr_labels)

        # XCY mode: out = [class_pred, attr0, attr1, ...]
        # independent/XC mode: out = [attr0, attr1, ...] (no class pred)
        attributes = out if self.is_independent else out[1:]

        out = attributes[self.out_index]

        # Now ensure it's a tensor
        assert isinstance(out, torch.Tensor), "Selected output must be a tensor"
        return out


class WrapperProtoCBM(nn.Module):
    """
    Wrap a model that returns a list into one that returns a tensor.
    You can choose which element (or combine them) for attribution.
    """
    def __init__(self, model, attr_labels=None):
        super().__init__()
        self.model = model
        self.attr_labels = attr_labels

    def forward(self, x, attr_labels=None):
        cls, attr, aux = self.model(x, self.attr_labels)

        out = attr

        # Now ensure it's a tensor
        assert isinstance(out, torch.Tensor), "Selected output must be a tensor"
        return out