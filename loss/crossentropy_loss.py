import torch
import torch.nn as nn
from torch.nn import functional as F

class CrossEntropyLabelSmooth(nn.Module):
    """
    Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        :param inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
        :param targets: ground truth labels with shape (num_classes)
        :return:
        """
        log_probs = self.logsoftmax(inputs)  # [64,751]
        y = torch.zeros(log_probs.size())  # [64,751]

        index = targets.unsqueeze(1).data.cpu()  # [64,1]
        # print("target:",targets,targets.shape)
        # print("target.unsqueeze(1):",targets.unsqueeze(1).shape)
        # print("targets.unsqueeze(1).data",targets.unsqueeze(1).data.shape)

        targets = y.scatter_(1, index, 1)
        # scatter_(dim, index, src),将src中数据根据index中的索引按照dim的方向进行填充
        # scatter_(): https://blog.csdn.net/t20134297/article/details/105755817

        if self.use_gpu:
            targets = targets.cuda()

        targets = (1-self.epsilon)*targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss

class LabelSmoothCrossEntropy(nn.Module):
    """
    Equation:
    """
    def __init__(self, smoothing=0.1):
        super().__init__()
        assert smoothing < 1
        self.smoothing = smoothing
        self.confidence = 1-smoothing

    def forward(self, x, targets):
        los_prbs = F.log_softmax(x, dim=-1)
        nll_loss = -los_prbs.gather(dim=-1, index=targets.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)

        smooth_loss = -los_prbs.mean(dim=-1)

        loss = self.confidence * nll_loss +self.smoothing * smooth_loss
        return loss.mean()