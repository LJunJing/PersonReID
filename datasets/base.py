from PIL import Image, ImageFile
from torch.utils.data import Dataset
import os.path as osp
ImageFile.LOAD_TRUNCATED_IMAGES = True #可以解决一些图像报错的问题

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

class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        """
        返回dataset的长度
        """
        return len(self.dataset)

    def __getitem__(self, index):
        """
        定义如何获得数据，按下标获得
        """

        #----------获取dataset列表中下标为index的元素---------
        img_path, pid, camid, trackid = self.dataset[index]
        # print("pid:",pid)

        #-----------打开图像----------
        img = read_image(img_path)

        #----------对图像做一个预处理--------
        if self.transform is not None:
            img = self.transform(img) #对图片进行预处理

        #---------返回处理后的图像、行人身份、摄像头编号、以及图片存储的名称（例如0001_c1s1_001051_00.jpg）
        return img, pid, camid, trackid, img_path.split('/')[-1]

class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        """
        获得图像信息，返回身份数、图像数、摄像头数、视角数
        """
        pids, cams, tracks = [], [], []

        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]

        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        # print("pid:", pid)

        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)

        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self, train, query, gallery):
        raise NotImplementedError

#数据集的基本信息
class BaseImageDataset(BaseDataset):
    """
    继承BaseDataset类，重写 print_dataset_statistics 方法，打印数据集的基本信息。
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")