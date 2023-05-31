import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, is_first=False):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels*4)
        )
        self.shortcut = nn.Sequential()
        if is_first:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*4 ,kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels*4)
            )
    def forward(self, x):
        out = self.bottleneck(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

class ResNet50(nn.Module):
    def __init__(self, num_classes=751):
        super().__init__()
        self.in_channels = 64
        self.in_planes = 2048
        self.bottleneck = Bottleneck
        #第一层是单独的，没有残差块
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        #conv2
        self.conv2 = self._make_layer(self.bottleneck, 64, 3)

        #conv3
        self.conv3 = self._make_layer(self.bottleneck, 128, 4)

        #conv4
        self.conv4 = self._make_layer(self.bottleneck, 256, 6)

        #conv5
        self.conv5 = self._make_layer(self.bottleneck, 512, 3)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

        self.classifier = nn.Linear(self.in_planes, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck2.bias.requires_grad_(False)
        self.bottleneck2.apply(weights_init_kaiming)

    def _make_layer(self, block, out_channels, layer_num):
        layers = []
        flag = True  # 表示每个block的第一层
        for i in range(0, layer_num):
            layers.append(block(self.in_channels, out_channels, is_first=flag))
            flag = False  # 除第一层之外的所有层
            self.in_channels = out_channels * 4
        return nn.Sequential(*layers)

    def forward(self, x, label=None):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)  # out: [64,2048,56,56]

        # out = self.avgpool(out)
        # out = out.reshape(out.shape[0], -1)  # out: [64,2048]

        out = nn.functional.avg_pool2d(out, out.shape[2:4])
        out = out.view(out.shape[0], -1) #out:[64, 2048]

        out = self.bottleneck2(out)

        # out = self.fc(out)  # out: [64,751]

        if self.training:
            cls_score = self.classifier(out)
            return cls_score, out  # global feature for triplet loss
        else:
            return out


# res50 = ResNet50()
# print(res50)