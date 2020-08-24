import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
from utils import cosine_similarity


class VGG16(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG16, self).__init__()
        model = models.vgg16(pretrained=pretrained)
        self.features = model.features
        layers = list(model.classifier.children())[:-1]
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        # from 224x224 to 4096
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class VGG19(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG19, self).__init__()
        model = models.vgg19(pretrained=pretrained)
        self.features = model.features
        layers = list(model.classifier.children())[:-1]
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        # from 224x224 to 4096
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class ResNet50(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50, self).__init__()
        model = models.resnet50(pretrained=pretrained)
        layers = list(model.children())[:-1]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # from 224x224 to 2048
        x = self.model(x)
        return x.view(x.size(0), -1)

    def logits(self, x):
        return self.last_layer(x)


class SEResNet50(nn.Module):
    def __init__(self, pretrained=True):
        super(SEResNet50, self).__init__()
        import pretrainedmodels
        if pretrained:
            model = pretrainedmodels.se_resnet50()
        else:
            model = pretrainedmodels.se_resnet50(pretrained=None)
        layers = list(model.children())[:-1]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # from 224x224 to 2048
        x = self.model(x)
        return x.view(x.size(0), -1)


class LinearProjection(nn.Module):
    '''Linear projection'''
    def __init__(self, n_in, n_out):
        super(LinearProjection, self).__init__()
        self.fc_embed = nn.Linear(n_in, n_out, bias=True)
        self.bn1d = nn.BatchNorm1d(n_out)
        self._init_params()

    def forward(self, x):
        x = self.fc_embed(x)
        x = self.bn1d(x)
        return x

    def _init_params(self):
        nn.init.xavier_normal(self.fc_embed.weight)
        nn.init.constant(self.fc_embed.bias, 0)
        nn.init.constant(self.bn1d.weight, 1)
        nn.init.constant(self.bn1d.bias, 0)


class ConvNet(nn.Module):
    def __init__(self, backbone, embedding):
        super(ConvNet, self).__init__()
        self.backbone = backbone
        self.embedding = embedding

    def forward(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        return x


class ProxyNet(nn.Module):
    """ProxyNet"""
    def __init__(self, n_classes, dim,
                 proxies=None, L2=False):
        super(ProxyNet, self).__init__()
        self.n_classes = n_classes
        self.dim = dim

        self.proxies = nn.Embedding(n_classes, dim,
                                    scale_grad_by_freq=False)

        if proxies is None:
            self.proxies.weight = nn.Parameter(
                torch.randn(self.n_classes, self.dim),
                requires_grad=True)
        else:
            self.proxies.weight = nn.Parameter(proxies, requires_grad=False)

        if L2:
            self.normalize_proxies()

    def normalize_proxies(self):
        norm = self.proxies.weight.data.norm(p=2, dim=1)[:, None]
        self.proxies.weight.data = self.proxies.weight.data / norm

    def forward(self, y_true):
        proxies_y_true = self.proxies(Variable(y_true))
        return proxies_y_true


class ProxyLoss(nn.Module):
    def __init__(self, temperature=1.):
        super(ProxyLoss, self).__init__()

        self.temperature = temperature

    def forward(self, x, y, proxies):
        """Proxy loss

        Arguments:
            x (Tensor): batch of features
            y (LongTensor): corresponding instance
        """
        loss = self.softmax_embedding_loss(x, y, proxies)

        preds = self.predict(x, proxies)

        acc = (y == preds).type(torch.FloatTensor).mean()

        return loss.mean(), acc

    def softmax_embedding_loss(self, x, y, proxies):
        idx = torch.from_numpy(np.arange(len(x), dtype=np.int)).cuda()
        diff_iZ = cosine_similarity(x, proxies)

        numerator_ip = torch.exp(diff_iZ[idx, y] / self.temperature)
        denominator_ip = torch.exp(diff_iZ / self.temperature).sum(1) + 1e-8
        return - torch.log(numerator_ip / denominator_ip)

    def classify(self, x, proxies):
        idx = torch.from_numpy(np.arange(len(x), dtype=np.int)).cuda()
        diff_iZ = cosine_similarity(x, proxies)

        numerator_ip = torch.exp(diff_iZ[idx, :] / self.temperature)
        denominator_ip = torch.exp(diff_iZ / self.temperature).sum(1) + 1e-8

        probs = numerator_ip / denominator_ip[:, None]
        return probs

    def predict(self, x, proxies):
        probs = self.classify(x, proxies)
        return probs.max(1)[1].data
