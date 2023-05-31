import argparse
import logging
import os
import os.path as osp
import torch
import numpy as np
import random
import sys
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import time
from torch.cuda import amp
from utils.logger import setup_logger
from model.FFusion import FFusion_cnn, FFusion_deit, FFusion
from datasets.dataloader import dataloader
from model.backbones import ResNet50, ViT, DeiT
from loss.make_loss import make_loss
from solver import make_optimizer, make_scheduler
from utils import AverageMeter, R1_mAP_eval

torch.cuda.set_device(1)

"""
设置随机种子
"""
def set_seed(seed):
    # 在神经网络中，参数默认是进行随机初始化的。不同的初始化参数往往会导致不同的结果。
    # 当得到比较好的结果时我们通常希望这个结果是可以复现的，在pytorch中，通过设置全局随机数种子可以实现这个目的。
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子

    # 如果读取数据的过程采用了随机预处理(如RandomCrop、RandomHorizontalFlip等)，
    # 那么对python、numpy的随机数生成器也需要设置种子
    np.random.seed(seed)  # 用于np定义的数据
    random.seed(seed)  # 用于一般的列表

    cudnn.deterministic = True  # 将这个 flag 置为True的话，每次返回的卷积算法将是确定的，即默认算法.配合上设置 Torch 的随机种子为固定值的话，应该可以保证每次运行网络的时候相同输入的输出是固定的
    cudnn.benchmark = True  # 设置这个 flag 为True，可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。

def train(model, model_name, center_criterion, train_loader, val_loader, optimizer, optimizer_center, scheduler, loss_fn, num_query, local_rank):
    log_period = 100
    checkpoint_period = 10
    eval_period = 25
    epochs = 100

    # logger = logging.getLogger("PersonReid.train")
    logger = setup_logger("PersonReid.train")
    logger.info('start training')

    # _LOCAL_PROCESS_GROUP = None

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=True) #num_query=3368
    scaler = amp.GradScaler()

    #train
    for epoch in range(1, epochs+1):
        # print("epoch{}".format(epoch))
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)

        model.train()
        n_iter=0
        # print("-----------------------------------------训练------------------------------")
        for iter, (img, pid, target_cam, target_view) in enumerate(train_loader):
            # train_loader 得到的是 train_collate_fn 返回的
            optimizer.zero_grad()
            optimizer_center.zero_grad()

            img = img.to(device) # [32,3,224,224]
            target = pid.to(device) #[32]
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)

            with amp.autocast(enabled=True):
                score, feat = model(img, target)
                # print("score",score.shape) #[32,196,,751]
                # print("feat",feat.shape) #[32,196,384]
                # print("target",target.shape)
                loss = loss_fn(score, feat, target)  # score:[32,751], feat:[32,1024], target:[32]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
                # print("acc", acc)
            else:
                acc = (score.max(1)[1] == target).float().mean()
                # print("acc", acc)

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))
            n_iter = n_iter + 1
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            save_path = os.path.join('logs', '{}'.format(model_name))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                # print("make success")
            torch.save(model.state_dict(),
                       os.path.join(save_path, model_name+'_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            model.eval()
            # print("-----------------------------------------验证------------------------------")
            for iter, (img, pid, camid, camids, target_view, _) in enumerate(val_loader):
                # val_loader 得到的是 val_collate_fn 返回的
                with torch.no_grad():
                    img = img.to(device)
                    target_view = target_view.to(device)
                    feat = model(img)
                    evaluator.update((feat, pid, camid))
            cmc, mAP, _, _, _, _, _ = evaluator.compute()
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            torch.cuda.empty_cache()

if __name__=='__main__':
    set_seed(1234)  # 设置随机种子

    # 加载数据
    train_loader, val_loader, num_query, num_classes, cam_num, view_num = dataloader('market1501')
    # print("-------------------------------------数据加载成功----------------------------")

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FFusion_cnn(num_classes=num_classes).to(device)
    # model = FFusion_deit(num_classes=num_classes).to(device)
    # model = FFusion(num_classes=num_classes).to(device)
    model_name = model.name()

    # print("-------------------------------------模型加载成功----------------------------")

    # 定义损失函数
    loss_func, center_criterion = make_loss(num_classes=num_classes)

    # 定义优化器
    optimizer, optimizer_center = make_optimizer(model, center_criterion)

    # 定义调度器：动态调度，lr衰减机制
    scheduler = make_scheduler(optimizer)

    # 设置输出文件路径
    output_dir = os.path.join('logs', '{}'.format(model_name))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger = setup_logger("FeatureFusion", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(output_dir))

    # 训练
    train(model, model_name, center_criterion, train_loader, val_loader, optimizer, optimizer_center, scheduler, loss_func, num_query, 1)