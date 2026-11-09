import anytree
import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import check_exists, makedir_exist_ok, save, load
from .utils import download_url, extract_file, make_classes_counts, make_tree, make_flat_index
from config import cfg

class TinyImageNet(Dataset):
    data_name = 'TinyImageNet'
    # 官方下载地址
    file = [('http://cs231n.stanford.edu/tiny-imagenet-200.zip', '90528d7ca1a48142e341f4ae8d2d6441')]

    def __init__(self, root, split, transform=None):
        self.root = os.path.expanduser(root)
        self.split = split  # 'train' or 'test' (Tiny-ImageNet 用 val 作为测试)
        self.transform = transform

        if not check_exists(self.processed_folder):
            self.process()

        # 加载处理好的数据
        # Tiny-ImageNet 较大，如果内存不足，可以考虑在 __getitem__ 中读取图片
        # 但为了保持与你原始代码逻辑一致，这里依然采用一次性 load 进内存
        id, self.data, self.target = load(os.path.join(self.processed_folder, '{}.pt'.format(self.split)),
                                          mode='pickle')

        self.classes_counts = make_classes_counts(self.target)
        self.classes_to_labels, self.target_size = load(os.path.join(self.processed_folder, 'meta.pt'), mode='pickle')
        self.other = {'id': id}

        # 针对 TinyImageNet 的类别过滤逻辑 (与你原代码类似)
        if 'all_class_num' in cfg and cfg['all_class_num'] > 0:
            num_classes = cfg['all_class_num']
            target_np = np.array(self.target)
            mask = target_np < num_classes
            self.data = self.data[mask]
            self.other['id'] = self.other['id'][mask]
            self.target = target_np[mask].tolist()
            self.classes_counts = make_classes_counts(self.target)

    def __getitem__(self, index):
        # self.data 是 numpy array (N, 64, 64, 3)
        data, target = Image.fromarray(self.data[index]), torch.tensor(self.target[index])
        other = {k: torch.tensor(self.other[k][index]) for k in self.other}
        input = {**other, 'data': data, 'target': target}
        if self.transform is not None:
            input = self.transform(input)
        return input

    def __len__(self):
        return len(self.data)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()

        raw_path = os.path.join(self.raw_folder, 'tiny-imagenet-200')
        train_set, test_set, meta = self.make_data(raw_path)

        save(train_set, os.path.join(self.processed_folder, 'train.pt'), mode='pickle')
        save(test_set, os.path.join(self.processed_folder, 'test.pt'), mode='pickle')
        save(meta, os.path.join(self.processed_folder, 'meta.pt'), mode='pickle')

    def download(self):
        makedir_exist_ok(self.raw_folder)
        for (url, md5) in self.file:
            filename = os.path.basename(url)
            download_url(url, self.raw_folder, filename, md5)
            extract_file(os.path.join(self.raw_folder, filename))

    def make_data(self, path):
        # 1. 获取类别 WNID 列表并排序以确保 label 固定
        wnids = sorted(os.listdir(os.path.join(path, 'train')))
        wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

        # 2. 读取训练集
        train_data = []
        train_target = []
        print("Loading Tiny-ImageNet training set...")
        for wnid in wnids:
            img_dir = os.path.join(path, 'train', wnid, 'images')
            for img_name in os.listdir(img_dir):
                img_path = os.path.join(img_dir, img_name)
                with Image.open(img_path) as img:
                    # 确保是 RGB
                    train_data.append(np.array(img.convert('RGB')))
                    train_target.append(wnid_to_label[wnid])

        train_data = np.array(train_data) # (100000, 64, 64, 3)

        # 3. 读取验证集 (作为 test)
        test_data = []
        test_target = []
        print("Loading Tiny-ImageNet validation set (as test)...")
        val_ann_file = os.path.join(path, 'val', 'val_annotations.txt')
        with open(val_ann_file, 'r') as f:
            for line in f:
                parts = line.split('\t')
                img_name = parts[0]
                wnid = parts[1]
                img_path = os.path.join(path, 'val', 'images', img_name)
                with Image.open(img_path) as img:
                    test_data.append(np.array(img.convert('RGB')))
                    test_target.append(wnid_to_label[wnid])

        test_data = np.array(test_data) # (10000, 64, 64, 3)

        train_id = np.arange(len(train_data)).astype(np.int64)
        test_id = np.arange(len(test_data)).astype(np.int64)

        # 4. 构建 anytree 结构
        # 获取 WNID 到人类可读名称的映射
        wnid_to_name = {}
        with open(os.path.join(path, 'words.txt'), 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if parts[0] in wnid_to_label:
                    wnid_to_name[parts[0]] = parts[1].split(',')[0] # 取第一个义项

        classes = [wnid_to_name[wnid] for wnid in wnids]
        classes_to_labels = anytree.Node('U', index=[])
        for c in classes:
            make_tree(classes_to_labels, [c])

        target_size = make_flat_index(classes_to_labels)

        return (train_id, train_data, train_target), (test_id, test_data, test_target), (classes_to_labels, target_size)

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.transform.__repr__())
        return fmt_str