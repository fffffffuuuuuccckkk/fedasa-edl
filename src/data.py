import copy
import torch
import numpy as np
import models
from config import cfg
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from utils import collate, to_device
import torchaudio
import random
data_stats = {'MNIST': ((0.1307,), (0.3081,)), 'FashionMNIST': ((0.2860,), (0.3530,)),
              'CIFAR10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
              'CIFAR100': ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
              'SVHN': ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
              'STL10': ((0.4409, 0.4279, 0.3868), (0.2683, 0.2610, 0.2687)),
              'GSC': ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
              'TinyImageNet': ((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
              'GTSRB': ((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629)),
              'MIOTCD': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))}


def fetch_dataset(data_name):
    import datasets
    dataset = {}
    print('fetching data {}...'.format(data_name))
    root = './data/{}'.format(data_name)
    if data_name in ['MNIST', 'FashionMNIST']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', '
                               'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['train'].transform = datasets.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset['test'].transform = datasets.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    elif data_name in ['CIFAR10', 'CIFAR100']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', '
                               'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['train'].transform = datasets.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset['test'].transform = datasets.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    elif data_name in ['SVHN','GTSRB']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', '
                               'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['train'].transform = datasets.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset['test'].transform = datasets.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    elif data_name in ['STL10']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', '
                               'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['train'].transform = datasets.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(96, padding=12, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset['test'].transform = datasets.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    elif data_name == 'TinyImageNet':
        # 1. 初始化数据集
        # 假设你的 TinyImageNet 类已经放到了 datasets 模块下或者当前作用域
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', '
                               'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))

        # 2. 设置训练集变换 (针对 64x64 图像)
        dataset['train'].transform = datasets.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=8, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])

        # 3. 设置测试集变换
        dataset['test'].transform = datasets.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    elif data_name == 'MIOTCD':
        # 1. 初始化数据集
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', '
                               'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))

        # 2. 设置训练集变换 (适配 64x64)
        # 考虑到自动驾驶场景，保留水平翻转，由于是路侧视角，随机裁剪有助于增强局部特征
        dataset['train'].transform = datasets.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=8, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])

        # 3. 设置测试集变换
        dataset['test'].transform = datasets.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    elif data_name == 'GSC':
        # 获取统计数据 (均值/方差)
        mean, std = data_stats[data_name][0], data_stats[data_name][1]
        m = mean[0] if isinstance(mean, (list, tuple)) else mean
        s = std[0] if isinstance(std, (list, tuple)) else std

        # 1. 初始化数据集 (注意：transform 设为 None，因为我们要手动指定)
        # 不要用 eval，直接用类
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=None)'.format(data_name))
        dataset['test']  = eval('datasets.{}(root=root, split=\'test\', '
                               'transform=None)'.format(data_name))

        # 定义简单的 Wrapper，因为 Dataset 需要一个 callable
        processor = GSCSpectrogramProcessor(mean=m, std=s)

        class GSCSupervisedTransform:
            def __init__(self, mode): self.mode = mode
            def __call__(self, input):
                input['data'] = processor.process(input['data'], mode=self.mode)
                return input

        dataset['train'].transform = GSCSupervisedTransform(mode='weak') # 训练集用弱增强
        dataset['test'].transform  = GSCSupervisedTransform(mode='test') # 测试集不用增强
    else:
        raise ValueError('Not valid dataset name')
    print('data ready')
    return dataset


def input_collate(batch):
    if isinstance(batch[0], dict):
        output = {key: [] for key in batch[0].keys()}
        for b in batch:
            for key in b:
                output[key].append(b[key])
        return output
    else:
        return default_collate(batch)


def make_data_loader(dataset, tag, batch_size=None, shuffle=None, sampler=None, batch_sampler=None):
    data_loader = {}
    for k in dataset:
        _batch_size = cfg[tag]['batch_size'][k] if batch_size is None else batch_size[k]
        _shuffle = cfg[tag]['shuffle'][k] if shuffle is None else shuffle[k]
        if sampler is not None:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, sampler=sampler[k],
                                        pin_memory=True, num_workers=cfg['num_workers'], collate_fn=input_collate,
                                        worker_init_fn=np.random.seed(cfg['seed']))
        elif batch_sampler is not None:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_sampler=batch_sampler[k],
                                        pin_memory=True, num_workers=cfg['num_workers'], collate_fn=input_collate,
                                        worker_init_fn=np.random.seed(cfg['seed']))
        else:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, shuffle=_shuffle,
                                        pin_memory=True, num_workers=cfg['num_workers'], collate_fn=input_collate,
                                        worker_init_fn=np.random.seed(cfg['seed']))

    return data_loader


def split_dataset(dataset, num_users, data_split_mode):
    data_split = {}
    if data_split_mode == 'iid':
        data_split['train'] = iid(dataset['train'], num_users)
        data_split['test'] = iid(dataset['test'], num_users)
    elif 'non-iid' in cfg['data_split_mode']:
        data_split['train'] = non_iid(dataset['train'], num_users)
        data_split['test'] = non_iid(dataset['test'], num_users)
    else:
        raise ValueError('Not valid data split mode')
    return data_split


def iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    data_split, idx = {}, list(range(len(dataset)))
    for i in range(num_users):
        num_items_i = min(len(idx), num_items)
        data_split[i] = torch.tensor(idx)[torch.randperm(len(idx))[:num_items_i]].tolist()
        idx = list(set(idx) - set(data_split[i]))
    return data_split


def non_iid(dataset, num_users):
    target = torch.tensor(dataset.target)
    data_split_mode_list = cfg['data_split_mode'].split('-')
    data_split_mode_tag = data_split_mode_list[-2]
    if data_split_mode_tag == 'l':
        data_split = {i: [] for i in range(num_users)}
        shard_per_user = int(data_split_mode_list[-1])
        target_idx_split = {}
        shard_per_class = int(shard_per_user * num_users / cfg['target_size'])
        for target_i in range(cfg['target_size']):
            target_idx = torch.where(target == target_i)[0]
            num_leftover = len(target_idx) % shard_per_class
            leftover = target_idx[-num_leftover:] if num_leftover > 0 else []
            new_target_idx = target_idx[:-num_leftover] if num_leftover > 0 else target_idx
            new_target_idx = new_target_idx.reshape((shard_per_class, -1)).tolist()
            for i, leftover_target_idx in enumerate(leftover):
                new_target_idx[i] = new_target_idx[i] + [leftover_target_idx.item()]
            target_idx_split[target_i] = new_target_idx
        target_split = list(range(cfg['target_size'])) * shard_per_class
        target_split = torch.tensor(target_split)[torch.randperm(len(target_split))].tolist()
        target_split = torch.tensor(target_split).reshape((num_users, -1)).tolist()
        for i in range(num_users):
            for target_i in target_split[i]:
                idx = torch.randint(len(target_idx_split[target_i]), (1,)).item()
                data_split[i].extend(target_idx_split[target_i].pop(idx))
    elif data_split_mode_tag == 'd':
        beta = float(data_split_mode_list[-1])
        dir = torch.distributions.dirichlet.Dirichlet(torch.tensor(beta).repeat(num_users))
        min_size = 0
        required_min_size = 10
        N = target.size(0)
        while min_size < required_min_size:
            data_split = [[] for _ in range(num_users)]
            for target_i in range(cfg['target_size']):
                target_idx = torch.where(target == target_i)[0]
                proportions = dir.sample()
                proportions = torch.tensor(
                    [p * (len(data_split_idx) < (N / num_users)) for p, data_split_idx in zip(proportions, data_split)])
                proportions = proportions / proportions.sum()
                split_idx = (torch.cumsum(proportions, dim=-1) * len(target_idx)).long().tolist()[:-1]
                # split_idx = torch.tensor_split(target_idx, split_idx)
                try:
                    # 尝试使用新版本的 tensor_split (PyTorch >= 1.8.0)
                    split_idx = torch.tensor_split(target_idx, split_idx)
                except AttributeError:
                    # 如果没有 tensor_split，手动进行切片 (PyTorch < 1.8.0)
                    # split_idx 此时是分割点的索引列表
                    indices = [0] + split_idx + [len(target_idx)]
                    split_idx = [target_idx[indices[i]:indices[i+1]] for i in range(len(indices)-1)]
                data_split = [data_split_idx + idx.tolist() for data_split_idx, idx in zip(data_split, split_idx)]
            min_size = min([len(data_split_idx) for data_split_idx in data_split])
        data_split = {i: data_split[i] for i in range(num_users)}
    else:
        raise ValueError('Not valid data split mode tag')
    return data_split


def separate_dataset_1(dataset, idx):
    separated_dataset = copy.deepcopy(dataset)
    separated_dataset.data = [dataset.data[s] for s in idx]
    separated_dataset.target = [dataset.target[s] for s in idx]
    separated_dataset.other['id'] = list(range(len(separated_dataset.data)))
    return separated_dataset



def separate_dataset(dataset, idx):

    # 创建空对象（不调用__init__）
    separated_dataset = dataset.__class__.__new__(dataset.__class__)

    for k, v in dataset.__dict__.items():

        if k == "data":
            # data 浅拷贝 + 取索引
            separated_dataset.data = v[idx]

        else:
            try:
                # 尝试进行深度拷贝
                setattr(separated_dataset, k, copy.deepcopy(v))
            except Exception as e:
                print("-" * 50)
                print(f"检测到异常！属性名: {k}")
                print(f"值的类型: {type(v)}")
                print(f"错误信息: {e}")

                # 如果是列表，进一步检查其内部元素
                if isinstance(v, list):
                    print(f"列表长度: {len(v)}")
                    for i, item in enumerate(v):
                        print(f"  元素 [{i}] 类型: {type(item)}")
                        # 检查是否由于元素内部包含不可拷贝的对象（如 iterator）
                        try:
                            copy.deepcopy(item)
                        except:
                            print(f"  [!] 发现不可拷贝的元素位于索引 {i}")

                # 特别检查：是否有人误改了 list 的 append 方法
                if hasattr(v, 'append'):
                    print(f"v.append 的类型: {type(v.append)}")

                print("-" * 50)
                raise e # 打印完后重新抛出异常，保持程序中断逻辑

    # 更新 target
    separated_dataset.target = [dataset.target[s] for s in idx]

    # 更新 id
    separated_dataset.other['id'] = list(range(len(idx)))

    return separated_dataset


def separate_dataset_su(server_dataset, client_dataset=None, supervised_idx=None):
    if supervised_idx is None:
        if cfg['data_name'] in ['STL10']:
            if cfg['num_supervised'] == -1:
                supervised_idx = torch.arange(5000).tolist()
            else:
                target = torch.tensor(server_dataset.target)[:5000]
                num_supervised_per_class = cfg['num_supervised'] // cfg['target_size']
                supervised_idx = []
                for i in range(cfg['target_size']):
                    idx = torch.where(target == i)[0]
                    idx = idx[torch.randperm(len(idx))[:num_supervised_per_class]].tolist()
                    supervised_idx.extend(idx)
        else:
            if cfg['num_supervised'] == -1:
                supervised_idx = list(range(len(server_dataset)))
            else:
                target = torch.tensor(server_dataset.target)
                if cfg['seen_class_num']:
                    num_supervised_per_class = cfg['num_supervised'] // cfg['seen_class_num']#cfg['target_size']
                else:
                    num_supervised_per_class = cfg['num_supervised'] // cfg['target_size']

                supervised_idx = []
                for i in range(cfg['target_size']):
                    if (cfg['task'] == 'OSR' or cfg['seen_class_num']) and i==cfg['seen_class_num']:
                        print(i)
                        break
                    idx = torch.where(target == i)[0]
                    idx = idx[torch.randperm(len(idx))[:num_supervised_per_class]].tolist()
                    supervised_idx.extend(idx)
    # idx = list(range(len(server_dataset)))
    # unsupervised_idx = list(set(idx) - set(supervised_idx))
    # # --- 新增：处理 ssl_ratio 截取部分给 server ---
    # _server_unlabeled_dataset = None
    # if cfg.get('ssl_ratio', 0) > 0:
    #     # 使用随机种子确保划分可复现
    #     import numpy as np
    #     np.random.seed(cfg['seed'])
    #     np.random.shuffle(unsupervised_idx)

    #     num_ssl = int(len(unsupervised_idx) * cfg['ssl_ratio'])
    #     server_ssl_idx = unsupervised_idx[:num_ssl]
    #     unsupervised_idx = unsupervised_idx[num_ssl:] # 剩下的留给 client

    #     # 准备 server 的无标签数据（通常需要强弱增强，所以使用 FixTransform）
    #     source_ds = client_dataset if client_dataset is not None else server_dataset
    #     _server_unlabeled_dataset = separate_dataset(source_ds, server_ssl_idx)
    #     _server_unlabeled_dataset.transform = FixTransform(cfg['data_name'])
    # # --------------------------------------------
    # _server_dataset = separate_dataset(server_dataset, supervised_idx)
    # if client_dataset is None:
    #     _client_dataset = separate_dataset(server_dataset, unsupervised_idx)
    # else:
    #     _client_dataset = separate_dataset(client_dataset, unsupervised_idx)
    #     transform = FixTransform(cfg['data_name'])
    #     _client_dataset.transform = transform
    # --- 2. 获取无标签索引池并过滤为仅限已知类 ---
    all_idx = list(range(len(server_dataset)))
    # 初步的无标签池：所有样本减去已经被选为有标签的样本
    unsupervised_idx_pool = list(set(all_idx) - set(supervised_idx))

    # 核心修改：筛选出池中属于已知类的样本索引 (label < seen_class_num)
    target_all = torch.tensor(server_dataset.target)
    K = cfg['seen_class_num']

    # 仅保留真实标签在已知类范围内的索引
    unsupervised_idx_known = [i for i in unsupervised_idx_pool if target_all[i] < K]
    # 同时也保留那些完全排除在外的索引（可选，通常 client 还是需要全类别的）
    # 但根据您的要求，Server 的 unlabeled 必须是已知类

    # --- 3. 从已知类无标签池中截取 ssl_ratio 给 Server ---
    _server_unlabeled_dataset = None
    if cfg.get('ssl_ratio', 0) > 0:
        import numpy as np
        np.random.seed(cfg['seed'])
        # 仅在已知类无标签索引中进行打乱和选取
        np.random.shuffle(unsupervised_idx_known)

        num_ssl = int(len(unsupervised_idx_known) * cfg['ssl_ratio'])
        server_ssl_idx = unsupervised_idx_known[:num_ssl]

        # 为了不破坏后续 client 的数据，我们从 pool 中移除这部分被 server 拿走的
        # 现在的 client_dataset 将持有剩下的所有数据（包括已知类剩余和所有未知类）
        unsupervised_idx_for_client = list(set(unsupervised_idx_pool) - set(server_ssl_idx))

        source_ds = client_dataset if client_dataset is not None else server_dataset
        _server_unlabeled_dataset = separate_dataset(source_ds, server_ssl_idx)
        # 设置 Transform
        from data import FixTransform
        _server_unlabeled_dataset.transform = FixTransform(cfg['data_name'])
    else:
        unsupervised_idx_for_client = unsupervised_idx_pool

    # --- 4. 封装返回 ---
    _server_dataset = separate_dataset(server_dataset, supervised_idx)

    if client_dataset is None:
        _client_dataset = separate_dataset(server_dataset, unsupervised_idx_for_client)
    else:
        _client_dataset = separate_dataset(client_dataset, unsupervised_idx_for_client)
        from data import FixTransform
        _client_dataset.transform = FixTransform(cfg['data_name'])
    if _server_unlabeled_dataset is not None:
        return _server_dataset, _client_dataset, supervised_idx, _server_unlabeled_dataset
    else:
        return _server_dataset, _client_dataset, supervised_idx, None


def make_batchnorm_dataset_su_1(server_dataset, client_dataset):
    batchnorm_dataset = copy.deepcopy(server_dataset)
    batchnorm_dataset.data = batchnorm_dataset.data + client_dataset.data
    batchnorm_dataset.target = batchnorm_dataset.target + client_dataset.target
    batchnorm_dataset.other['id'] = batchnorm_dataset.other['id'] + client_dataset.other['id']
    return batchnorm_dataset


def make_batchnorm_dataset_su(server_dataset, client_dataset):

    # 创建空对象（避免 deepcopy 整个 dataset）
    batchnorm_dataset = server_dataset.__class__.__new__(server_dataset.__class__)

    for k, v in server_dataset.__dict__.items():

        if k == "data":
            # data 浅拷贝 + 拼接
            batchnorm_dataset.data = np.concatenate(
                (server_dataset.data, client_dataset.data), axis=0
            )

        else:
            # 其它字段 deepcopy
            setattr(batchnorm_dataset, k, copy.deepcopy(v))

    # target 拼接
    batchnorm_dataset.target = server_dataset.target + client_dataset.target

    # id 拼接
    batchnorm_dataset.other['id'] = (
        server_dataset.other['id'] + client_dataset.other['id']
    )

    return batchnorm_dataset

def make_batchnorm_dataset_su_1(server_dataset, client_dataset):
    # 1. 使用浅拷贝创建 Server 数据集的副本
    batchnorm_dataset = copy.copy(server_dataset)

    # 2. 列表加法会返回一个包含两部分引用的新列表，非常安全
    batchnorm_dataset.data = server_dataset.data + client_dataset.data
    batchnorm_dataset.target = server_dataset.target + client_dataset.target

    # 3. 同样需要浅拷贝字典以防污染
    if hasattr(server_dataset, 'other') and isinstance(server_dataset.other, dict):
        batchnorm_dataset.other = server_dataset.other.copy()
        batchnorm_dataset.other['id'] = server_dataset.other['id'] + client_dataset.other['id']

    return batchnorm_dataset


def make_dataset_normal(dataset):
    import datasets
    _transform = dataset.transform
    if cfg['data_name'] == 'GSC':
        # 1. 获取统计参数
        mean, std = data_stats['GSC'][0], data_stats['GSC'][1]
        m = mean[0] if isinstance(mean, (list, tuple)) else mean
        s = std[0] if isinstance(std, (list, tuple)) else std

        # 2. 实例化处理器 (确保 GSCSpectrogramProcessor 在 data.py 中已定义)
        # 如果你之前没在 data.py 定义该类，请确保该类可见，或者在这里重新定义
        processor = GSCSpectrogramProcessor(mean=m, std=s)

        # 3. 定义一个简单的适配器，因为 datasets.Compose 里的 transform 接收 data 返回 data
        class GSCStatTransform:
            def __call__(self, waveform):
                # 使用 'test' 模式 (不加噪声，只转谱图)
                return processor.process(waveform, mode='test')

        # 4. 设置 transform
        # 这里只放一个转换器，它完成了 Audio -> Image -> Normalize 的全过程
        transform = datasets.Compose([GSCStatTransform()])
    else:
        transform = datasets.Compose([transforms.ToTensor(), transforms.Normalize(*data_stats[cfg['data_name']])])
    dataset.transform = transform
    return dataset, _transform


def make_batchnorm_stats(dataset, model, tag):
    with torch.no_grad():
        test_model = copy.deepcopy(model)
        test_model.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=True))
        dataset, _transform = make_dataset_normal(dataset)
        data_loader = make_data_loader({'train': dataset}, tag, shuffle={'train': False})['train']
        test_model.train(True)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input = to_device(input, cfg['device'])
            # print(input.keys())
            # print(input['target'].shape)
            # import sys
            # sys.exit()
            if (cfg['task'] == 'OSR' or cfg['seen_class_num']) and 'target' in input:
                # 创建一个与原始 target 形状相同，且值全部为 -1 的 Tensor
                # 保持原始 target 的 dtype，通常是 torch.long

                # 方法一：使用 torch.ones_like() 然后乘以 -1
                # input['target'] = -1 * torch.ones_like(input['target'])

                # 方法二：使用 fill_
                input['target'].fill_(-1)
            test_model(input)
        dataset.transform = _transform
    return test_model


class FixTransform(object):
    def __init__(self, data_name):
        import datasets
        self.is_audio = False
        if data_name in ['CIFAR10', 'CIFAR100']:
            self.weak = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name])
            ])
            self.strong = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                datasets.RandAugment(n=2, m=10),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name])
            ])
        elif data_name in ['SVHN','GTSRB']:
            self.weak = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name])
            ])
            self.strong = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                datasets.RandAugment(n=2, m=10),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name])
            ])
        elif data_name in ['STL10']:
            self.weak = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(96, padding=12, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name])
            ])
            self.strong = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(96, padding=12, padding_mode='reflect'),
                datasets.RandAugment(n=2, m=10),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name])
            ])
        elif data_name == 'TinyImageNet':
            self.weak = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(64, padding=8, padding_mode='reflect'), # 64x64 图像
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name])
            ])
            self.strong = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(64, padding=8, padding_mode='reflect'),
                datasets.RandAugment(n=2, m=10), # FixMatch 核心：RandAugment
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name])
            ])
        elif data_name == 'MIOTCD':
            self.weak = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                # 针对 64x64 图像，使用 padding=8 是标准做法（1/8 比例）
                transforms.RandomCrop(64, padding=8, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name])
            ])
            self.strong = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(64, padding=8, padding_mode='reflect'),
                # FixMatch 核心：RandAugment。对于 64 像素的真实场景图像，n=2, m=10 是很稳健的配置
                datasets.RandAugment(n=2, m=10),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name])
            ])
        elif data_name == 'GSC':
            # 获取统计数据
            mean, std = data_stats[data_name][0], data_stats[data_name][1]
            m = mean[0] if isinstance(mean, (list, tuple)) else mean
            s = std[0] if isinstance(std, (list, tuple)) else std

            # 初始化我们自定义的处理器
            self.gsc_processor = GSCSpectrogramProcessor(mean=m, std=s)

            # 标记这是一个音频任务，不需要构建 torchvision.transforms.Compose
            self.is_audio = True
        else:
            raise ValueError('Not valid dataset')

    def __call__(self, input):
        if self.is_audio:
            # 输入 input['data'] 是原始音频波形 (numpy array)
            waveform = input['data']

            # 1. 生成弱增强视图 (data)
            weak_view = self.gsc_processor.process(waveform, mode='weak')

            # 2. 生成强增强视图 (aug)
            strong_view = self.gsc_processor.process(waveform, mode='strong')

            # 3. 更新字典
            # 注意：现在的 data 和 aug 已经是 [3, 32, 32] 的 Tensor 了
            input = {**input, 'data': weak_view, 'aug': strong_view}
            return input
        data = self.weak(input['data'])
        aug = self.strong(input['data'])
        input = {**input, 'data': data, 'aug': aug}
        return input

class GSCSpectrogramProcessor:
    def __init__(self, mean=0.0, std=1.0):
        # 注意：对于原始音频波形，通常数据已经在 [-1, 1] 之间
        # 默认 mean=0, std=1 表示不做额外归一化，或者你可以填入计算好的数据集均值方差
        self.mean = mean
        self.std = std
        # 这里不再需要 torchaudio.transforms.MelSpectrogram 等组件

    def process(self, waveform, mode='weak'):
        # --- 0. 格式清理 ---
        # 确保输入是 tensor
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.from_numpy(waveform)

        # 确保维度是 [Channels, Length]，即 [1, 16000]
        # 1D CNN 通常需要 input shape 为 (N, C, L)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # --- 1. 时域增强 (Time-Domain Augmentation) ---

        if mode == 'weak':
            # 弱增强：轻微的时间平移 (Shift)
            # 800 samples = 0.05s (at 16kHz)
            shift = random.randint(-800, 800)
            waveform = torch.roll(waveform, shifts=shift, dims=-1)

        elif mode == 'strong':
            # 强增强：大幅平移 + 加噪 + 幅度缩放

            # 1. 大幅平移
            shift = random.randint(-2000, 2000) # 0.125s
            waveform = torch.roll(waveform, shifts=shift, dims=-1)

            # 2. 加高斯白噪声 (Additive White Gaussian Noise)
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise

            # 3. 幅度随机缩放 (Amplitude Scale/Gain)
            # 模拟不同录音设备的音量差异
            scale = random.uniform(0.8, 1.2)
            waveform = waveform * scale

        # mode == 'test' 时直接跳过增强，原样输出

        # --- 2. 归一化 (Normalization) ---
        # 这一步将数据标准化，有助于模型收敛
        # 注意：不需要像图片那样做 MinMax 到 0-1，保留正负值对波形很重要
        waveform = (waveform - self.mean) / self.std

        # 返回形状: [1, 16000]
        return waveform

class MixDataset(Dataset):
    def __init__(self, size, dataset):
        self.size = size
        self.dataset = dataset

    def __getitem__(self, index):
        index = torch.randint(0, len(self.dataset), (1,)).item()
        input = self.dataset[index]
        input1 = {'data': input['data'], 'target': input['target']}
        if cfg['open_close_ratio']:
            input1['aug'] = input['aug']
        return input1

    def __len__(self):
        return self.size


