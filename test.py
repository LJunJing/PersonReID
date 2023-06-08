import os
import torch
import torch.nn as nn
from datasets import dataloader
from model.FFusion import FFusion, FFusion_cnn, FFusion_deit
from utils import R1_mAP_eval
from utils.logger import setup_logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(1)

def test(model, val_loader, num_query):
    logger = setup_logger("PersonReid.test")
    logger.info('start testing')

    # feat_norm：测试前特征是否正常化，如果是，则相当于余弦距离
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=True)
    evaluator.reset()  # 清空

    # if device:
    #     if torch.cuda.device_count() > 1:
    #         print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
    #         model = nn.DataParallel(model)  # 数据并行的方法
    #     model.to(device)
    model.to(device)
    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img)  # [64,1024]
            # print("feat", feat.shape)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]

if __name__ == "__main__":
    # global all_rank_5, all_rank_1

    # 加载数据
    train_loader, val_loader, num_query, num_classes, cam_num, view_num = dataloader('market1501')

    # -----------加载模型----------------
    # model = FFusion_cnn(num_classes=num_classes)
    model = FFusion_deit(num_classes=num_classes)
    # model = FFusion(num_classes=num_classes)
    model_name = model.name()
    model.load_param('logs/{}/{}_100.pth'.format(model_name, model_name)) #加载预训练好的参数

    # -----------输出文件----------
    output_dir = os.path.join('logs_test', '{}'.format(model_name))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger = setup_logger("FeatureFusion", output_dir, if_train=False)
    logger.info("Saving model in the path :{}".format(output_dir))

    test(model, val_loader, num_query)
