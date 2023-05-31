import torch

def make_optimizer(model, center_criterion):
    params = []

    # enumerate(model.parameters())返回: index,param
    # model.name_parameters()返回: name,param
    # name即网络层的weight或bias，例如fc1.weight, fc1.bias。
    # param就是weight、bias等对应的值
    # 例如print(name,param.shape), 得到conv1.weight torch.Size([32, 3, 3, 3])
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lr = 3e-4
        weight_decay = 0.0005
        params += [{"params": [param], "lr": lr, "weight_decay": weight_decay}]
        # params返回一个列表，列表中每个元素都是一个字典，字典中有三个键值对，分别是params、lr、weight_decay

    optimizer = getattr(torch.optim, 'Adam')(params)  # params是给Adam传的参数
    # getattr: 从对象中获取属性。getattr(x,'y')相当于x.y
    # 当给定默认参数时，该属性不存在时返回默认参数。若不给定默认参数，当属性不存在时会引发异常。

    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=0.5)
    return optimizer, optimizer_center
    # 共返回两个优化器，一个是Adam优化器，一个是SGD作为中心优化器
    # Optimizer
