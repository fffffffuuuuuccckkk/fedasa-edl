import anytree
import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import check_exists, makedir_exist_ok, save, load
from .utils import extract_file, make_classes_counts, make_tree, make_flat_index
from config import cfg

class MIOTCD(Dataset):
    data_name = 'MIOTCD'
    
    def __init__(self, root, split, transform=None):
        self.root = os.path.expanduser(root)
        self.split = split  # 'train' 或 'test'
        self.transform = transform
        
        if not check_exists(self.processed_folder):
            self.process()
        
        # 加载处理后的数据 (.pt 文件)
        id, self.data, self.target = load(os.path.join(self.processed_folder, '{}.pt'.format(self.split)),
                                         mode='pickle')
        
        self.classes_counts = make_classes_counts(self.target)
        self.classes_to_labels, self.target_size = load(os.path.join(self.processed_folder, 'meta.pt'), mode='pickle')
        self.other = {'id': id}

        # 标签过滤逻辑
        if 'all_class_num' in cfg and cfg['all_class_num'] > 0:
            num_classes = cfg['all_class_num']
            target_np = np.array(self.target)
            mask = target_np < num_classes
            self.data = self.data[mask]
            self.other['id'] = self.other['id'][mask]
            self.target = target_np[mask].tolist()
            self.classes_counts = make_classes_counts(self.target)

    def __getitem__(self, index):
        img_array = self.data[index]
        data = Image.fromarray(img_array)
        target = torch.tensor(self.target[index])
        
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
            raise RuntimeError(f"Dataset not found in {self.raw_folder}.")
        
        # 如果 raw 下没有 train1 文件夹，则尝试解压
        train1_path = os.path.join(self.raw_folder, 'train1')
        if not os.path.exists(train1_path):
            archive_path = os.path.join(self.raw_folder, 'archive.zip')
            if os.path.exists(archive_path):
                print("Extracting archive.zip...")
                extract_file(archive_path, self.raw_folder)
            else:
                raise RuntimeError("Neither 'train1' folder nor 'archive.zip' found.")

        train_set, test_set, meta = self.make_data()
        
        makedir_exist_ok(self.processed_folder)
        save(train_set, os.path.join(self.processed_folder, 'train.pt'), mode='pickle')
        save(test_set, os.path.join(self.processed_folder, 'test.pt'), mode='pickle')
        save(meta, os.path.join(self.processed_folder, 'meta.pt'), mode='pickle')

    def make_data(self):
        # 路径指向你截图中的 train1
        data_path = os.path.join(self.raw_folder, 'train1')
        
        # 获取类别文件夹列表
        classes = sorted(entry.name for entry in os.scandir(data_path) if entry.is_dir())
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        all_images = []
        all_labels = []

        print(f"Reading images from {data_path}...")
        for cls_name in classes:
            cls_dir = os.path.join(data_path, cls_name)
            files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # 如果样本过多导致内存崩溃，可以在此处对每个类取采样：files = files[:5000]
            for img_name in files:
                img_path = os.path.join(cls_dir, img_name)
                img = Image.open(img_path).convert('RGB')
                img = img.resize((64, 64), Image.BILINEAR)
                all_images.append(np.array(img))
                all_labels.append(class_to_idx[cls_name])
        
        all_images = np.array(all_images)
        all_labels = np.array(all_labels)

        # 核心：因为 Kaggle 没给 test，所以手动划分 (80% train, 20% test)
        num_samples = len(all_images)
        indices = np.arange(num_samples)
        np.random.seed(42) # 固定随机种子保证实验可重复
        np.random.shuffle(indices)
        
        split_point = int(num_samples * 0.8)
        train_idx = indices[:split_point]
        test_idx = indices[split_point:]

        # 整理成 (id, data, target) 格式对齐 load()
        train_data, train_target = all_images[train_idx], all_labels[train_idx].tolist()
        test_data, test_target = all_images[test_idx], all_labels[test_idx].tolist()
        
        train_id = np.arange(len(train_data)).astype(np.int64)
        test_id = np.arange(len(test_data)).astype(np.int64)

        # 构建 anytree 结构（保持层级索引一致）
        classes_to_labels = anytree.Node('U', index=[])
        for c in classes:
            make_tree(classes_to_labels, [c])
        target_size = make_flat_index(classes_to_labels)

        return (train_id, train_data, train_target), (test_id, test_data, test_target), (classes_to_labels, target_size)

    def __repr__(self):
        return f'Dataset {self.__class__.__name__}\nSize: {self.__len__()}\nRoot: {self.root}\nSplit: {self.split}'