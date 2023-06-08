import os
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import glob
import os.path as osp
import torchvision.transforms as transformers
from PIL import Image
from torchvision import transforms
import streamlit as st
from model.FFusion import FFusion, FFusion_cnn
from datasets.market1501 import Market1501
from utils.logger import setup_logger
torch.cuda.set_device(1)

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    global img
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

# def infer_collate_fn(batch):
#     imgs, pids, camids, viewids, img_paths = zip(*batch)
#     viewids = torch.tensor(viewids, dtype=torch.int64)
#     camids_batch = torch.tensor(camids, dtype=torch.int64)
#     return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths

# def infer_dataset(dataset_dir):
#     query_dir = 'data/inference/query'
#     gallery_dir = 'data/inference/gallery'
#
#     pid_begin = 1
#
#
# def infer_dataloader(query, gallery):
#     infer_transforms = transformers.Compose([
#         # transformers.Resize([224, 224]),  # ResNet50
#         transformers.Resize([256, 128]),  # DeiT
#         transformers.ToTensor(),
#         transformers.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#
#     num_workers = 8
#     num_query = len(query)
#     infer_set = ImageDataset(query+gallery, infer_transforms)
#
#     # ---------- 采样器 -----------
#     # print('using softmax sampler')
#
#     infer_loader = DataLoader(infer_set, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=infer_collate_fn)
#
#     return infer_loader, num_query

def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    # return dist_mat.cpu().numpy()
    return dist_mat.detach().cpu().numpy()

# def test(model, query, gallery, num_query):
#     logger = setup_logger("PersonReid.test")
#     logger.info('start inference')
#
#     # feat_norm：测试前特征是否正常化，如果是，则相当于余弦距离
#     # evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=True)
#     # evaluator.reset()  # 清空
#
#     # if device:
#     #     if torch.cuda.device_count() > 1:
#     #         print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
#     #         model = nn.DataParallel(model)  # 数据并行的方法
#     #     model.to(device)
#
#     infer_transforms = transformers.Compose([
#                 # transformers.Resize([224, 224]),  # ResNet50
#                 transformers.Resize([256, 128]),  # DeiT
#                 transformers.ToTensor(),
#                 transformers.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#             ])
#
#     model.eval()
#
#     # 读取query图片，并将其进行预处理
#     query_img = infer_transforms(read_image(query))
#     query = np.array(query_img)
#     print("query", query.size)
#
#     # 输入网络，获得query特征
#     query_feat = model(query_img)
#     print("query_feat", query_feat.shape)
#
#     gallerys = glob.glob(osp.join(gallery, '*.jpg'))
#     distance = []
#     for img in gallerys:
#         with torch.no_grad():
#             img = infer_transforms(read_image(img)).to(device)
#             gallery_feat = model(img)
#             distmat = euclidean_distance(query_feat, gallery_feat)
#             print("distmat", distmat)
#
#             if distmat > 0.7:
#                 distance.append(distmat)
#
#     # logger.info("Validation Results ")
#     # logger.info("mAP: {:.1%}".format(mAP))
#     # for r in [1, 5, 10]:
#     #     logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
#     # return cmc[0], cmc[4]

def softmax(f):
    # instead: first shift the values of f so that the highest number is 0:
    f -= np.max(f) # f becomes [-666, -333, 0]
    return np.exp(f) / np.sum(np.exp(f))

def get_img(fileDir, num, tarDir=None):
    pathDir = os.listdir(fileDir) #获得图片的原始路径
    # file_num = len(pathDir) # 数据总量
    # img_num = int(file_num * rate)
    sample = random.sample(pathDir, num) #随机选取img_num数量的样本图片
    print(sample)
    if tarDir:
        for name in sample:
            shutil.copy(fileDir+name, tarDir+name)
    return sample

def del_files(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        # 判断是否是一个目录,若是,则递归删除
        if os.path.isdir(f_path):
            del_files(f_path)
        else:
            os.remove(f_path)

def checkDir(file):
    if os.path.exists(file):
        sz = os.path.getsize(file)
        if sz:
            del_files(file)
    else:
        os.makedirs(file)

class ST:
    def __init__(self):
        self.dataset = Market1501()
        self.img_size = [384, 128]
        self.to_PIL = transforms.Compose([
            # transforms.Normalize(mean= -0.5 / 0.5, std = 1 / 0.5),
            transforms.ToPILImage(),
        ])

        st.title('Pedestrian Re-identification')
        st.sidebar.title('Options')

        if 'uploaded_img' not in st.session_state: #标记query图是上传图像还是随机采样图像，默认为随机采样
            st.session_state['uploaded_img'] = False

    def show(self):
        with st.container():
            st.write("This is inside the container")
            query_img_column, gallery_img_column = st.columns(2)

            self.query_img = query_img_column.image(np.zeros([self.img_size, 4], dtype=float), width=self.img_width, caption='query image')
            query_img_column.buttton('Random', on_click=self.random_img) #点击random按钮执行random_img方法

            self.upload_img = query_img_column.file_uploaeder('Upload', key=0, on_change=self.uploaded_img)

            query = random.sample(self.dataset.query_dir, 1)
            if self.upload_img is not None and st.session_state['uploaded_img']:

                self.query_img.image(Image.open(self.upload_img), width=self.img_size, caption='query image')
            else:
                self.query_img.image(self.to_PIL(self.norm_range(query)), width=self.img_size, caption='original image')

        st.write("This is outside the container")


    def uploaded_img(self):
        st.session_state['uploaded_img'] = True

    def random_img(self):
        st.session_state['uploaded_img'] = False
        # self.ori_img.image()
        # query = self.dataset.query_dir
        # query_img = random.sample(query, 1)
        # self.ori_img.image(query_img)
        st.write("open a random image")

    def _norm_ip(self, img, low, high):
        img = img.clamp_(min=low, max=high)
        return img.sub_(low).div_(max(high - low, 1e-5))

    def norm_range(self, t):
        return self._norm_ip(t, float(t.min()), float(t.max()))

if __name__ == "__main__":
    # # -----------输出文件----------
    # output_dir = "results"
    # if output_dir and not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # logger = setup_logger("FeatureFusion", output_dir, if_train=False)
    # logger.info("Saving model in the path :{}".format("results"))

    # -----------加载模型----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FFusion_cnn(num_classes=751)
    # model = FFusion_deit(num_classes=751)
    # model = FFusion(num_classes=751)
    model_name = model.name()
    model.load_param('logs/{}/{}_100.pth'.format(model_name, model_name)) #加载预训练好的参数
    model.to(device)
    model.eval()

    # # -----------加载数据----------------
    # query = 'data/inference/query/query.jpg'  # 接收输入进来的query图片，保存在data/inference/query中
    # num_query = len(query)
    # # original = ['original.jpg']  # 接收输入进来的包含多个人的未进行行人检测的图片
    # # gallery = detection_net(before_detect)  # 输入到检测网络中获得行人图像，这是一个列表或set，作为gallery集，保存在data/inference/gallery中
    # gallery = 'data/inference/gallery'
    #
    # test(model, query, gallery, num_query)

    infer_transforms = transformers.Compose([
        # transformers.Resize([224, 224]),  # ResNet50
        transformers.Resize([256, 128]),  # DeiT
        transformers.ToTensor(),
        transformers.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_name = 'market1501'  # 需获取，字符串

    queryDir = 'data/{}/query/'.format(dataset_name)
    query_tarDir = 'data/inference/query/'
    checkDir(query_tarDir)

    galleryDir = 'data/{}/bounding_box_test/'.format(dataset_name)
    gallery_tarDir = 'data/inference/gallery/'
    checkDir(gallery_tarDir)

    # query_path = 'data/inference/query/query.jpg'
    query_path = get_img(queryDir, 1, query_tarDir)
    query = read_image(query_tarDir+query_path[0])  # 需展示，图片
    query = infer_transforms(query).unsqueeze(0).to(device)  # [1,3,256,128]
    query_feat = model(query) # [1,1024]
    # print("query_feat", query_feat.shape)

    gallery_num = 5  # 需获取，整数类型，预先定义为5
    gallery_rate = get_img(galleryDir, gallery_num, gallery_tarDir)
    gallery_path = glob.glob(osp.join('data/inference/gallery', '*.jpg'))

    distance = []
    dis2img = {}
    gallery_feat = []
    for idx, img_path in enumerate(gallery_path):
        with torch.no_grad():
            gallery = read_image(img_path)  # 需展示，图片
            gallery = infer_transforms(gallery).unsqueeze(0).to(device)
            gallery_feat = model(gallery)  # [1,1024]
            # print(type(gallery))
            # gallery_feats.append(gallery_feat.cpu())
            distmat = euclidean_distance(query_feat, gallery_feat) #将distmat展示在相应图片下方，浮点
            print(distmat, img_path[-23:])

    # gallery_feats = torch.tensor(gallery_feat)
    # gallery_feat = torch.cat(gallery_feats, dim=0)
    # distmat = euclidean_distance(query_feat, gallery_feat)  # numpy.ndarray




