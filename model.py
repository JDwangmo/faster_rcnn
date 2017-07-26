import torch.nn as nn
from roi_pooling import roi_pooling as _roi_pooling
from torchvision import models
from rpn import RPN as _RPN
from faster_rcnn import FasterRCNN as _FasterRCNN


class _Features(nn.Module):
    def __init__(self):
        super(_Features, self).__init__()

        vgg16 = models.vgg16(pretrained=True)
        # print vgg16.features
        # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg16_feature = nn.Sequential(*list(vgg16.features.children())[:29])
        # in_channels=3, out_channels=3, kernel_size=3, stride=16, padding=1
        # 3x3x3x3
        # self.m = nn.Conv2d(3, 3, 3, 16, 1)
        self.m = self.vgg16_feature

    def forward(self, x):
        return self.m(x)


class _Classifier(nn.Module):
    def __init__(self):
        super(_Classifier, self).__init__()
        self.m1 = nn.Linear(3 * 7 * 7, 21)
        self.m2 = nn.Linear(3 * 7 * 7, 21 * 4)

    def forward(self, x):
        return self.m1(x), self.m2(x)


def _pooler(x, rois):
    # mini-batch , C, 7, 7
    x = _roi_pooling(x, rois, size=(7, 7), spatial_scale=1.0 / 16.0)
    # mini-batch , C * 7 * 7
    return x.view(x.size(0), -1)


class _RPNClassifier(nn.Module):
    def __init__(self, n):
        super(_RPNClassifier, self).__init__()
        self.m1 = nn.Conv2d(n, 18, 3, 1, 1)
        self.m2 = nn.Conv2d(n, 36, 3, 1, 1)

    def forward(self, x):
        return self.m1(x), self.m2(x)


def model():
    _features = _Features()
    _classifier = _Classifier()
    _rpn_classifier = _RPNClassifier(3)

    _rpn = _RPN(
        classifier=_rpn_classifier
    )
    _model = _FasterRCNN(
        features=_features,
        pooler=_pooler,
        classifier=_classifier,
        rpn=_rpn
    )
    return _model
