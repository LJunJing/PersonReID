import torch
import torchvision.transforms as transformers
from torch.utils.data import DataLoader
from timm.data.random_erasing import RandomErasing

from .base import ImageDataset
from .market1501 import Market1501

__factory = {
    'market1501': Market1501,
    # 'dukemtmc': DukeMTMCreID,
    # 'msmt17': MSMT17,
    # 'occ_duke': OCC_DukeMTMCreID,
    # 'veri': VeRi,
    # 'VehicleID': VehicleID,
}

def train_collate_fn(batch):
    """
    collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids , _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths

def dataloader(dataset_name):
    # ----------对输入图片做一系列转换---------------
    # ---------- 数据预处理 ----------
    train_transforms = transformers.Compose([
        # transformers.Resize([224, 224], interpolation=3),  # ResNet50
        transformers.Resize([256, 128], interpolation=3), #DeiT
        transformers.RandomHorizontalFlip(p=0.5),
        transformers.Pad(10),
        # transformers.RandomCrop([224, 224]),
        transformers.RandomCrop([256, 128]),
        transformers.ToTensor(),
        transformers.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        RandomErasing(probability=0.5, mode='pixel', max_count=1, device='cpu')
    ])

    val_transforms = transformers.Compose([
        # transformers.Resize([224, 224]),  # ResNet50
        transformers.Resize([256, 128]), #DeiT
        transformers.ToTensor(),
        transformers.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #----------设置线程数-----------------
    num_workers = 8

    #---------加载数据集-------------------
    dataset = __factory[dataset_name]()
    num_classes = dataset.num_train_pids  # 获得身份类别总数，751
    cam_num = dataset.num_train_cams  # 获得摄像头数
    view_num = dataset.num_train_vids  # 获得视角数

    train_set = ImageDataset(dataset.train, train_transforms)
    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

    # ---------- 采样器 -----------
    # print('using softmax sampler')

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=num_workers, collate_fn=train_collate_fn)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=num_workers, collate_fn=val_collate_fn)

    return train_loader, val_loader, len(dataset.query), num_classes, cam_num, view_num
