# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re
import os.path as osp
from .base import BaseImageDataset
from collections import defaultdict
import pickle

class Market1501(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'market1501'

    def __init__(self, root='data', verbose=True, pid_begin = 0, **kwargs):
        super(Market1501, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        # 判断路径是否存在
        self._check_before_run()

        self.pid_begin = pid_begin

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery) #打印数据集的基本信息

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """
        Check if all files are available before going deeper
        检查路径是否存在
        """
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        #----------读取jpg图像---------------------
        # glob模块用来查找文件目录和文件，并将搜索到的结果返回到一个列表中，该list中包含dir_path中所有.jpg格式的图片
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))

        #----------定义匹配模式----------------
        # 由于数据集的文件命名规则为"0001_c1s1_001051_00.jpg"，表示：行人身份_摄像头编号&拍摄到的序列的编号_帧数号_边框号
        # 因此正则匹配式([-\d]+)_c(\d)，能够匹配并获取行人身份号和摄像头编号，得到的结果为：(行人ID号，摄像机编号),即0001_c1
        pattern = re.compile(r'([-\d]+)_c(\d)')

        #------------行人身份set-----------
        pid_container = set()
        for img_path in sorted(img_paths):  #sorted(img_paths)返回一个新列表，其中包含按升序排列的img_paths中的所有项
            pid, _ = map(int, pattern.search(img_path).groups())
            # 正则表达式中，group方法用来提出分组截获的字符串。
            # group()等价于group(0)，返回匹配到的整体结果，即0001_c1；group(1)表示正则表达式pattern中第一个()匹配到的部分，即0001;group(2)以此类推。
            # map将匹配到的部分映射成整数，得到('0001','1')
            # pid==0001==1

            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid) #将此行人身份加入到set集中
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        #-------------数据集------------
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel:#如果relabel=True
                pid = pid2label[pid]

            dataset.append((img_path, self.pid_begin + pid, camid, 1))

        return dataset
        # 返回的dataset是一个列表，列表中的每一项的格式为（图片路径，行人身份，摄像头编号，1）
        # 其中，行人身份从1开始，最大为1501；摄像头编号从0开始，最大为5
        # 如('data/market1501/query/0001_c1s1_001051_00.jpg', 1, 0, 1)


