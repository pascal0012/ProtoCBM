
import torch
import torch.nn as nn

class WrapperCUB(nn.Module):
    """
    Wrap a model that returns a list into one that returns a tensor.
    You can choose which element (or combine them) for attribution.
    """
    def __init__(self, model, out_index=0):
        super().__init__()
        self.model = model
        self.out_index = out_index

    def forward(self, x):
        out = self.model(x)
        _class_pred, attributes = out[0], out[1:]

        # If model returns a list, select one element or concatenate them
        if isinstance(out, (list, tuple)):
            out = attributes[self.out_index]

        # Now ensure it's a tensor
        assert isinstance(out, torch.Tensor), "Selected output must be a tensor"
        return out
    

class WrapperProtoCBM(nn.Module):
    """
    Wrap a model that returns a list into one that returns a tensor.
    You can choose which element (or combine them) for attribution.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        cls, attr, aux = self.model(x)
        
        out = attr

        # Now ensure it's a tensor
        assert isinstance(out, torch.Tensor), "Selected output must be a tensor"
        return out