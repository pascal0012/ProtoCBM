import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F


#################
###   MODEL   ###
#################

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class resnet_proto_IoU(nn.Module):
    """
        Hardcoded to the CUB dataset. 
    """
    def __init__(self, resnet_path:str, avg_pool=False, train=False):
        super(resnet_proto_IoU, self).__init__()
        resnet = models.resnet101()
        num_ftrs = resnet.fc.in_features
        num_fc = 150
        resnet.fc = nn.Linear(num_ftrs, num_fc)

        state_dict = torch.load(resnet_path)
        resnet.load_state_dict(state_dict)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fine_tune(train)

        # 02 - load cls weights
        # we left the entry for several layers, but here we only use layer4
        self.dim_dict = {'layer1': 56*56, 'layer2': 28*28, 'layer3': 14*14, 'layer4': 7*7, 'avg_pool': 1*1}
        self.channel_dict = {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048, 'avg_pool': 2048}
        self.kernel_size = {'layer1': 56, 'layer2': 28, 'layer3': 14, 'layer4': 7, 'avg_pool': 1}
        self.extract = ['layer4']  # 'layer1', 'layer2', 'layer3', 'layer4'
        self.epsilon = 1e-4

        self.softmax = nn.Softmax(dim=1)
        self.softmax2d = nn.Softmax2d()
        self.sigmoid = nn.Sigmoid()
        self.prototype_vectors = dict()
        for name in self.extract:
            prototype_shape = [312, self.channel_dict[name], 1, 1]
            self.prototype_vectors[name] = nn.Parameter(2e-4 * torch.rand(prototype_shape), requires_grad=True)
        self.prototype_vectors = nn.ParameterDict(self.prototype_vectors)
        self.ALE_vector = nn.Parameter(2e-4 * torch.rand([312, 2048, 1, 1]), requires_grad=True)
        self.avg_pool = avg_pool

    def forward(self, x, return_map=False):
        """out: predict class, predict attributes, maps, out_feature"""
        record_features = {}
        batch_size = x.size(0)
        x = self.resnet[0:5](x)  # layer 1
        record_features['layer1'] = x  # [64, 256, 56, 56]
        x = self.resnet[5](x)  # layer 2
        record_features['layer2'] = x  # [64, 512, 28, 28]
        x = self.resnet[6](x)  # layer 3
        record_features['layer3'] = x  # [64, 1024, 14, 14]
        x = self.resnet[7](x)  # layer 4
        record_features['layer4'] = x  # [64, 2048, 7, 7]

        attention = dict()
        pre_attri = dict()
        pre_class = dict()

        if self.avg_pool:
            pre_attri['final'] = F.avg_pool2d(F.conv2d(input=x, weight=self.ALE_vector), kernel_size=7).view(batch_size, -1)
        else:
            pre_attri['final'] = F.max_pool2d(F.conv2d(input=x, weight=self.ALE_vector), kernel_size=7).view(batch_size, -1)
        output_final = self.softmax(pre_attri['final'])
        name = self.extract[-1]
        attention = F.conv2d(input=record_features[name], weight=self.prototype_vectors[name])  # [64, 312, W, H]
        pre_attri = F.max_pool2d(attention, kernel_size=self.kernel_size[name]).view(batch_size, -1)
        # pre_class = self.softmax(pre_attri[name].mm(attribute))
        return output_final, pre_attri, attention

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


def load_apn_baseline(args, train=False):
    model = resnet_proto_IoU(args.backbone_dir, train=train)
    print("Any operations on the classification output of APN is invalid (e.g. accuracy computation)!")
    return model
