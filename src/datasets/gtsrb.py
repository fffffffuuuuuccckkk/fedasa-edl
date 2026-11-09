import anytree
import numpy as np
import os
import csv
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import check_exists, makedir_exist_ok, save, load
from .utils import download_url, extract_file, make_classes_counts, make_tree, make_flat_index
from config import cfg

class GTSRB(Dataset):
    data_name = 'GTSRB'
    # GTSRB 通常手动下载，这里保留 file 结构以匹配风格，md5 设为 None 或实际值
    file = [('Train.tar', None), ('Test.tar', None), ('Test.csv', None)]

    def __init__(self, root, split, transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform

        if not check_exists(self.processed_folder):
            self.process()

        id, self.data, self.target = load(os.path.join(self.processed_folder, '{}.pt'.format(self.split)),
                                          mode='pickle')

        self.classes_counts = make_classes_counts(self.target)
        self.classes_to_labels, self.target_size = load(os.path.join(self.processed_folder, 'meta.pt'), mode='pickle')
        self.other = {'id': id}

        # 模仿你的 label 调整逻辑
        if cfg.get('adjust_classes') and cfg.get('data_name') == 'GTSRB':
            # 示例：如果是为了某些实验需要重排 43 个类别
            print(f"GTSRB: Adjusting labels for split '{split}'")
            self.target = [(label + 1) % 43 for label in self.target]

        # 类别过滤逻辑
        if 'all_class_num' in cfg and cfg['all_class_num'] > 0:
            num_classes = cfg['all_class_num']
            target_np = np.array(self.target)
            mask = target_np < num_classes
            self.data = self.data[mask]
            self.other['id'] = self.other['id'][mask]
            self.target = target_np[mask].tolist()
            self.classes_counts = make_classes_counts(self.target)

    def __getitem__(self, index):
        # GTSRB 原始图片大小不一，在 make_data 时已统一 resize 为 32x32 或按需调整
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
        # 根据你的路径，raw 应该就在 GTSRB 目录下
        return self.root

    def process(self):
        # 如果 raw 下有 tar 文件但没解压，调用 extract_file
        for filename, _ in self.file:
            file_path = os.path.join(self.raw_folder, filename)
            if filename.endswith('.tar') and os.path.exists(file_path):
                print(f"Extracting {filename}...")
                extract_file(file_path)

        train_set, test_set, meta = self.make_data()

        makedir_exist_ok(self.processed_folder)
        save(train_set, os.path.join(self.processed_folder, 'train.pt'), mode='pickle')
        save(test_set, os.path.join(self.processed_folder, 'test.pt'), mode='pickle')
        save(meta, os.path.join(self.processed_folder, 'meta.pt'), mode='pickle')

    def make_data(self):
        img_size = (32, 32)

        # --- 1. 自动定位训练集路径 ---
        # 尝试几种可能的解压路径结构
        possible_train_base = [
            os.path.join(self.raw_folder, 'Train'),
            os.path.join(self.raw_folder, 'train'),
            os.path.join(self.raw_folder, 'GTSRB', 'Train'),
            self.raw_folder # 某些解压方式直接把 00000 放在 raw 下
        ]

        train_path = None
        for p in possible_train_base:
            # 检查是否存在第 0 类文件夹 (尝试 00000 或 0 两种格式)
            if os.path.exists(os.path.join(p, '00000')) or os.path.exists(os.path.join(p, '0')):
                train_path = p
                break

        if train_path is None:
            raise FileNotFoundError(f"无法在 {self.raw_folder} 及其子目录下找到 GTSRB 训练数据文件夹 (00000 或 0)")

        # --- 2. 处理训练集 ---
        train_data = []
        train_target = []
        print(f"Processing GTSRB Train set from: {train_path}")

        for c in range(43):
            # 同时兼容 '00000' 和 '0' 两种文件夹命名格式
            class_path = os.path.join(train_path, str(c).zfill(5))
            if not os.path.exists(class_path):
                class_path = os.path.join(train_path, str(c))

            if not os.path.exists(class_path):
                print(f"Warning: Class {c} directory not found, skipping.")
                continue

            # 找到文件夹下所有图片 (兼容 .ppm, .png, .jpg)
            imgs = [f for f in os.listdir(class_path) if f.lower().endswith(('.ppm', '.png', '.jpg'))]
            for img_name in imgs:
                img_path = os.path.join(class_path, img_name)
                try:
                    with Image.open(img_path) as img:
                        train_data.append(np.array(img.convert('RGB').resize(img_size)))
                        train_target.append(c)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")

        # --- 3. 自动定位测试集路径 ---
        possible_test_base = [
            os.path.join(self.raw_folder, 'Test'),
            os.path.join(self.raw_folder, 'test'),
            os.path.join(self.raw_folder, 'GTSRB', 'Test'),
            self.raw_folder
        ]

        test_path = None
        for p in possible_test_base:
            # 测试集通常包含大量图片，检查是否有图片文件
            if os.path.exists(p) and any(f.lower().endswith(('.ppm', '.png')) for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))):
                test_path = p
                break

        # 寻找 Test.csv
        test_csv = os.path.join(self.raw_folder, 'Test.csv')
        if not os.path.exists(test_csv):
            # 尝试在 Test 目录下找
            test_csv = os.path.join(test_path if test_path else "", 'Test.csv')

        # --- 4. 处理测试集 ---
        test_data = []
        test_target = []
        if test_path and os.path.exists(test_csv):
            print(f"Processing GTSRB Test set from: {test_path}")
            with open(test_csv, 'r') as f:
                # 兼容分号 ; 或 逗号 , 分隔符
                first_line = f.readline()
                f.seek(0)
                sep = ';' if ';' in first_line else ','
                reader = csv.DictReader(f, delimiter=sep)
                for row in reader:
                    # 获取文件名并去除可能存在的路径前缀
                    img_name = os.path.basename(row['Path'])
                    img_real_path = os.path.join(test_path, img_name)
                    if os.path.exists(img_real_path):
                        with Image.open(img_real_path) as img:
                            test_data.append(np.array(img.convert('RGB').resize(img_size)))
                            test_target.append(int(row['ClassId']))
        else:
            print("Warning: Test set or Test.csv not found, test set will be empty.")

        # --- 5. 封装返回 ---
        train_data = np.array(train_data)
        test_data = np.array(test_data)
        train_id = np.arange(len(train_data)).astype(np.int64)
        test_id = np.arange(len(test_data)).astype(np.int64)

        # 构建 anytree (保持原逻辑)
        classes = [f'Class{i}' for i in range(43)]
        classes_to_labels = anytree.Node('U', index=[])
        for i in range(43):
            if i <= 20: group = 'Prohibitory_Danger'
            elif i <= 31: group = 'Mandatory'
            else: group = 'Others'
            make_tree(classes_to_labels, [group, classes[i]])

        target_size = make_flat_index(classes_to_labels, classes)
        return (train_id, train_data, train_target), (test_id, test_data, test_target), (classes_to_labels, target_size)

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.transform.__repr__())
        return fmt_str