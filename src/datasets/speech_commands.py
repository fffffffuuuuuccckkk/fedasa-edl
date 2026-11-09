import os
import torch
import torchaudio
import numpy as np
import anytree
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from utils import check_exists, makedir_exist_ok, save, load
from .utils import download_url, extract_file, make_classes_counts, make_tree, make_flat_index
from config import cfg

class GSC(Dataset):
    data_name = 'GSC'
    # 使用 v0.01 版本 (65k 样本，最接近 CIFAR-10)
    file = [('http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz', '743935421bb51cccdb6bdd152e04c5c70274e935c82119ad7faeec31780d811d')]

    # 核心关键词 (10类)，其余归为 Unknown
    CORE_LABELS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

    def __init__(self, root, split, transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        if not check_exists(self.processed_folder):
            self.process()

        # 加载处理好的数据 (完全模仿 CIFAR10 的结构)
        id, self.data, self.target = load(os.path.join(self.processed_folder, '{}.pt'.format(self.split)), mode='pickle')
        self.classes_counts = make_classes_counts(self.target)
        self.classes_to_labels, self.target_size = load(os.path.join(self.processed_folder, 'meta.pt'), mode='pickle')
        self.other = {'id': id}

    def __getitem__(self, index):
        # --- 修改点 1: 直接返回 float32 的波形数据，不要转 PIL Image ---
        data = self.data[index].astype(np.float32)
        target = torch.tensor(self.target[index])
        other = {k: torch.tensor(self.other[k][index]) for k in self.other}
        # input['data'] 现在是 shape=(16000,) 的 numpy 数组
        input = {**other, 'data': data, 'target': target}

        # 注意：这里的 transform 会在 data.py 里定义，负责 Audio -> Image 的转换
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

        print("Processing Speech Commands audio to Spectrograms (this may take a while)...")
        train_set, test_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, 'train.pt'), mode='pickle')
        save(test_set, os.path.join(self.processed_folder, 'test.pt'), mode='pickle')
        save(meta, os.path.join(self.processed_folder, 'meta.pt'), mode='pickle')

    def download(self):
        makedir_exist_ok(self.raw_folder)
        for (url, md5) in self.file:
            filename = os.path.basename(url)
            download_url(url, self.raw_folder, filename, md5)
            # 解压到 raw_folder/speech_commands_v0.01
            extract_file(os.path.join(self.raw_folder, filename), os.path.join(self.raw_folder, 'speech_commands_v0.01'))

    def make_data(self):
        data_path = os.path.join(self.raw_folder, 'speech_commands_v0.01')
        all_folders = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)) and d != '_background_noise_']
        classes = sorted(all_folders)

        data_list = []
        target_list = []

        print("Processing Audio Waveforms...")
        for label_idx, label_name in enumerate(tqdm(classes)):
            label_dir = os.path.join(data_path, label_name)
            for wav_file in os.listdir(label_dir):
                if not wav_file.endswith('.wav'): continue

                # 加载音频
                waveform, sr = torchaudio.load(os.path.join(label_dir, wav_file))

                # --- 修改点 2: 统一处理成 16000 长度的 1D 向量 ---
                if waveform.shape[1] < 16000:
                    waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]))
                else:
                    waveform = waveform[:, :16000]

                # 保存为 numpy float32 (节省空间且精度够用)
                data_list.append(waveform.squeeze(0).numpy())
                target_list.append(label_idx)

        # 转换为 numpy 数组
        data_arr = np.array(data_list, dtype=np.float32)
        target_arr = np.array(target_list)

        # --- 修改点 3: 划分逻辑不变，直接返回 ---
        perm = np.random.permutation(len(target_arr))
        split_idx = int(len(target_arr) * 0.85)
        train_idx, test_idx = perm[:split_idx], perm[split_idx:]

        train_data, train_target = data_arr[train_idx], target_arr[train_idx].tolist()
        test_data, test_target = data_arr[test_idx], target_arr[test_idx].tolist()

        train_id = np.arange(len(train_data)).astype(np.int64)
        test_id = np.arange(len(test_data)).astype(np.int64)

        classes_to_labels = anytree.Node('U', index=[])
        for c in classes:
            make_tree(classes_to_labels, [c])
        target_size = make_flat_index(classes_to_labels)

        return (train_id, train_data, train_target), (test_id, test_data, test_target), (classes_to_labels, target_size)

import torch
import torch.nn as nn
import torchaudio.transforms as T
import random

class AudioToSpectrogramPipe(nn.Module):
    def __init__(self, mode='test'):
        super().__init__()
        self.mode = mode # 'weak', 'strong', 'test'

        # 1. 基础转换器: Audio -> MelSpec
        # 参数调整为输出接近 32x32 的尺寸
        self.melspec = T.MelSpectrogram(
            sample_rate=16000,
            n_mels=32,      # 对应图像高度 H=32
            n_fft=1024,
            hop_length=500  # 16000/500 ≈ 32，对应图像宽度 W=32
        )
        self.db = T.AmplitudeToDB()

    def forward(self, waveform):
        # 输入 waveform: numpy array (16000,) or Tensor
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.from_numpy(waveform)

        # 确保是 (1, 16000)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # --- A. 时域增强 (Audio Augmentation) ---
        if self.mode in ['weak', 'strong']:
            # 1. Time Shift (简单的 roll 操作)
            shift = random.randint(-1600, 1600) # 偏移 0.1秒
            waveform = torch.roll(waveform, shifts=shift, dims=-1)

            # 2. Add Noise (仅 Strong)
            if self.mode == 'strong':
                noise = torch.randn_like(waveform) * 0.005
                waveform = waveform + noise

        # --- B. 转频域 (Convert to Image) ---
        spec = self.melspec(waveform) # -> [1, 32, 32]
        spec = self.db(spec)

        # --- C. 频域增强 (SpecAugment) ---
        if self.mode == 'strong':
            # Time Masking
            spec = T.TimeMasking(time_mask_param=5)(spec)
            # Frequency Masking
            spec = T.FrequencyMasking(freq_mask_param=5)(spec)

        # --- D. 尺寸与归一化适配 (Adapt to CIFAR format) ---
        # 1. 强制 Resize 到 32x32 (消除 hop_length 的取整误差)
        spec = torch.nn.functional.interpolate(spec.unsqueeze(0), size=(32, 32), mode='bilinear', align_corners=False).squeeze(0)

        # 2. 归一化 (简单做 Instance Norm，让数值分布在类似图像的区间)
        # 因为后续 data.py 里通常不建议再做 Normalize，或者这里输出后可以直接由 data_stats 处理
        # 这里我们把谱图归一化到 roughly [0, 1] 区间，方便后续 ToTensor 逻辑（虽然已经是 Tensor 了）
        spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)

        # 3. 复制为 3 通道 [1, 32, 32] -> [3, 32, 32] 以适配 ResNet
        spec = spec.repeat(3, 1, 1)

        return spec