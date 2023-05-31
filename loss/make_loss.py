from .crossentropy_loss import CrossEntropyLabelSmooth
from torch.nn import functional as F
from .center_loss import CenterLoss
from .triplet_loss import TripletLoss

def make_loss(num_classes):
    # sampler = 'softmax_triplet'

    # triplet = TripletLoss(0.3)  # triplet loss
    # print("using triplet loss with margin:{}".format(0.3))

    # cross_entropy = CrossEntropyLabelSmooth(num_classes=num_classes)
    # print("label smooth on, numclasses:", num_classes)

    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=2048, use_gpu=True)

    def loss_func(score, feat, target):
        return F.cross_entropy(score, target)

    # def loss_func(score, feat, target):
    #     # score:[64,751], feat:[64,2048], target:[64]
    #     if isinstance(score, list):
    #         ID_LOSS = [cross_entropy(scor, target) for scor in score[1:]]
    #         ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
    #         ID_LOSS = 0.5 * ID_LOSS + 0.5 * cross_entropy(score[0], target)
    #     else:
    #         ID_LOSS = cross_entropy(score, target)
    #
    #     if isinstance(feat, list):
    #         TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
    #         TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
    #         TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
    #     else:
    #         TRI_LOSS = triplet(feat, target)[0]
    #
    #     return ID_LOSS + TRI_LOSS

    return loss_func, center_criterion
