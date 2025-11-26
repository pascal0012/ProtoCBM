import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import Parameter

from components import BasicConv2d, FC, InceptionA, InceptionB, InceptionC, InceptionD, InceptionE


BACKBONE_URLS = {
    "inception_v3": "https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth",
}

class Inception3(nn.Module):
    def __init__(
        self,
        aux_logits=True,
        n_attributes=0,
        expand_dim=0,
        pretrained=True,
        freeze=True
    ):
        """
        Args:
            aux_logits: whether to also output auxiliary logits
            n_attributes: number of attributes to predict
            expand_dim: if not 0, add an additional fc layer with expand_dim neurons
            pretrained: whether we should load model weights
        """
        super(Inception3, self).__init__()

        self.aux_logits = aux_logits
        self.n_attributes = n_attributes
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        if aux_logits:
            self.AuxLogits = InceptionAux(
                768,
                n_attributes=self.n_attributes,
                expand_dim=expand_dim,
            )
        self.final_channel_dim = 2048
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(self.final_channel_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats

                stddev = m.stddev if hasattr(m, "stddev") else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Loading in model, if pretrained
        if pretrained:
            self.load_partial_state_dict(model_zoo.load_url(BACKBONE_URLS["inception_v3_google"]))
            if freeze:  # only finetune fc layer
                for name, param in self.named_parameters():
                    if "fc" not in name: 
                        param.requires_grad = False

    def forward(self, x):
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)  # N x 192 x 35 x 35
        x = self.Mixed_5b(x)  # N x 256 x 35 x 35
        x = self.Mixed_5c(x)  # N x 288 x 35 x 35
        x = self.Mixed_5d(x)  # N x 288 x 35 x 35
        x = self.Mixed_6a(x)  # N x 768 x 17 x 17
        x = self.Mixed_6b(x)  # -> N x 768 x 17 x 17
        x = self.Mixed_6c(x)  # -> N x 768 x 17 x 17
        x = self.Mixed_6d(x)  # -> N x 768 x 17 x 17
        x = self.Mixed_6e(x)  # -> N x 768 x 17 x 17

        if self.training and self.aux_logits:
            out_aux = self.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)  # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)  # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)  # N x 2048 x 8 x 8

        if self.training and self.aux_logits:
            return x, out_aux
        else:
            return x

    def load_partial_state_dict(self, state_dict):
        """
        If dimensions of the current model doesn't match the pretrained one (esp for fc layer), load whichever weights that match
        """
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state or "fc" in name:
                continue
            if isinstance(param, Parameter):
                param = param.data
            own_state[name].copy_(param)


class InceptionAux(nn.Module):
    def __init__(
        self,
        in_channels,
        n_attributes=0,
        expand_dim=0,
    ):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.n_attributes = n_attributes
        self.expand_dim = expand_dim

        self.all_fc = nn.ModuleList()
        for i in range(self.n_attributes):
            self.all_fc.append(FC(768, 1, expand_dim, stddev=0.001))


    def forward(self, x):
        # N x 768 x 17 x 17
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 768 x 1 x 1
        x = x.view(x.size(0), -1)
        # N x 768
        out = []
        for fc in self.all_fc:
            out.append(fc(x))
        return out
