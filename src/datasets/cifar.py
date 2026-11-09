import anytree
import numpy as np
import os
import pickle
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import check_exists, makedir_exist_ok, save, load
from .utils import download_url, extract_file, make_classes_counts, make_tree, make_flat_index
from config import cfg

class CIFAR10(Dataset):
    data_name = 'CIFAR10'
    file = [('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', 'c58f30108f718f92721af3b95e74349a')]

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
        if cfg['adjust_classes'] and cfg['data_name'] == 'CIFAR10':
            print(f"CIFAR-10: Adjusting labels for split '{split}' during initialization.")
            # # 定义映射规则
            # label_map = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 0, 7: 8, 8: 2, 9: 9}
            # label_map = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 0: 6, 1: 7, 8: 8, 9: 9}
            # # 使用列表推导式高效地创建一个新的、已映射的标签列表
            # # self.target 是一个 numpy 数组或列表，可以直接迭代
            # self.target = [label_map[label] for label in self.target]
            self.target = [(label - 2) % 10 for label in self.target]
        if cfg['adjust_classes'] and cfg['data_name'] == 'CIFAR100':
            super_classes = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                                  3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                                  6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                                  0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                                  5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                                  16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                                  10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                                  2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                                  16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                                  18, 1, 2, 15, 6, 0, 17, 8, 14, 13])
            # 1. 获取排序索引：根据超类 ID 对原始类别 ID 进行排序
            # argsort 会返回一个索引数组，表示原标签应该如何排列才能按照超类顺序组合在一起
            # 例如：所有属于超类0的类排在最前面，然后是超类1的类...
            new_order = np.argsort(super_classes)

            # 2. 生成映射字典: {旧标签: 新标签}
            # new_order[0] 是原来某个标签，现在它变成了新标签 0
            # new_order[1] 是原来某个标签，现在它变成了新标签 1
            label_map = {old_label: new_label for new_label, old_label in enumerate(new_order)}

            # 3. 应用映射到 self.target
            # 这样，self.target 中的 0-99 就会被重排，使得同一超类的样本拥有相邻的标签 ID
            self.target = [label_map[label] for label in self.target]

        if 'all_class_num' in cfg and cfg['all_class_num'] > 0:
            num_classes = cfg['all_class_num']
            # print(f"Filtering dataset: keeping only the first {num_classes} classes.")

            # 将 target 转为 numpy array 以便使用布尔索引
            target_np = np.array(self.target)

            # 生成掩码 (Mask): 保留标签小于 num_classes 的样本
            mask = target_np < num_classes

            # 应用掩码进行过滤
            self.data = self.data[mask]
            self.other['id'] = self.other['id'][mask]
            self.target = target_np[mask].tolist() # 转回 list 保持格式一致

            # (可选) 更新类别统计，因为数据变了
            self.classes_counts = make_classes_counts(self.target)
        # =============== 新增: OOD 白噪声替换逻辑 ===============
        if cfg.get('client_noise_ood', False):
            target_np = np.array(self.target)

            # 找到标签为 9 的所有样本（即 CIFAR-10 的最后一类）
            mask = (target_np == 9)
            num_noise = np.sum(mask)

            if num_noise > 0:
                # 给定 CIFAR-10 的均值和方差
                # 设定形状为 (1, 1, 1, 3) 方便向 (num_noise, 32, 32, 3) 的数据进行广播计算
                mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 1, 1, 3)
                std = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 1, 1, 3)

                # 生成和原类别图像相同数量的高斯白噪声，生成范围大约在 [0, 1] 左右浮动
                noise = np.random.normal(loc=mean, scale=std, size=(num_noise, 32, 32, 3))

                # 因为原始图像存储为 uint8 类型且范围在 [0, 255]
                # 这里将生成的噪声数据映射到 0~255 并截断，以适应图像像素空间
                noise = np.clip(noise * 255.0, 0, 255).astype(np.uint8)

                # 直接将对应标签的图片数据替换为生成的白噪声
                self.data[mask] = noise
        # ========================================================


    def __getitem__(self, index):
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
        train_set, test_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, 'train.pt'), mode='pickle')
        save(test_set, os.path.join(self.processed_folder, 'test.pt'), mode='pickle')
        save(meta, os.path.join(self.processed_folder, 'meta.pt'), mode='pickle')
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        for (url, md5) in self.file:
            filename = os.path.basename(url)
            download_url(url, self.raw_folder, filename, md5)
            extract_file(os.path.join(self.raw_folder, filename))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.transform.__repr__())
        return fmt_str

    def make_data(self):
        train_filenames = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        test_filenames = ['test_batch']
        train_data, train_target = read_pickle_file(os.path.join(self.raw_folder, 'cifar-10-batches-py'),
                                                    train_filenames)
        test_data, test_target = read_pickle_file(os.path.join(self.raw_folder, 'cifar-10-batches-py'), test_filenames)
        train_id, test_id = np.arange(len(train_data)).astype(np.int64), np.arange(len(test_data)).astype(np.int64)
        with open(os.path.join(self.raw_folder, 'cifar-10-batches-py', 'batches.meta'), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            classes = data['label_names']
        classes_to_labels = anytree.Node('U', index=[])
        for c in classes:
            make_tree(classes_to_labels, [c])
        target_size = make_flat_index(classes_to_labels)
        return (train_id, train_data, train_target), (test_id, test_data, test_target), (classes_to_labels, target_size)


class CIFAR100(CIFAR10):
    data_name = 'CIFAR100'
    file = [('https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz', 'eb9058c3a382ffc7106e4002c42a8d85')]

    def make_data(self):
        train_filenames = ['train']
        test_filenames = ['test']
        train_data, train_target = read_pickle_file(os.path.join(self.raw_folder, 'cifar-100-python'), train_filenames)
        test_data, test_target = read_pickle_file(os.path.join(self.raw_folder, 'cifar-100-python'), test_filenames)
        train_id, test_id = np.arange(len(train_data)).astype(np.int64), np.arange(len(test_data)).astype(np.int64)
        with open(os.path.join(self.raw_folder, 'cifar-100-python', 'meta'), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            classes = data['fine_label_names']
        classes_to_labels = anytree.Node('U', index=[])
        for c in classes:
            for k in CIFAR100_classes:
                if c in CIFAR100_classes[k]:
                    c = [k, c]
                    break
            make_tree(classes_to_labels, c)
        target_size = make_flat_index(classes_to_labels, classes)
        return (train_id, train_data, train_target), (test_id, test_data, test_target), (classes_to_labels, target_size)


def read_pickle_file(path, filenames):
    img, label = [], []
    for filename in filenames:
        file_path = os.path.join(path, filename)
        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            img.append(entry['data'])
            label.extend(entry['labels']) if 'labels' in entry else label.extend(entry['fine_labels'])
    img = np.vstack(img).reshape(-1, 3, 32, 32)
    img = img.transpose((0, 2, 3, 1))
    return img, label


CIFAR100_classes = {
    'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    'food containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
    'fruit and vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    'household electrical devices': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
    'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'large man-made outdoor things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'people': ['baby', 'boy', 'girl', 'man', 'woman'],
    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
    'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    'vehicles 2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
}
