import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import init_param, make_batchnorm, loss_fn
from config import cfg

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equal_inout = (in_planes == out_planes)
        self.shortcut = (not self.equal_inout) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                             padding=0, bias=False) or None

    def forward(self, x):
        if not self.equal_inout:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equal_inout else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        out = torch.add(x if self.equal_inout else self.shortcut(x), out)
        return out


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate):
        super().__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer(x)
        return out


class WideResNet(nn.Module):
    def __init__(self, data_shape, num_classes, depth, widen_factor, drop_rate):
        super().__init__()
        num_down = int(min(math.log2(data_shape[1]), math.log2(data_shape[2]))) - 3
        hidden_size = [16]
        for i in range(num_down + 1):
            hidden_size.append(16 * (2 ** i) * widen_factor)
        n = ((depth - 1) / (num_down + 1) - 1) / 2
        block = BasicBlock
        blocks = []
        blocks.append(nn.Conv2d(data_shape[0], hidden_size[0], kernel_size=3, stride=1, padding=1, bias=False))
        blocks.append(NetworkBlock(n, hidden_size[0], hidden_size[1], block, 1, drop_rate))
        for i in range(num_down):
            blocks.append(NetworkBlock(n, hidden_size[i + 1], hidden_size[i + 2], block, 2, drop_rate))
        blocks.append(nn.BatchNorm2d(hidden_size[-1]))
        blocks.append(nn.ReLU(inplace=True))
        blocks.append(nn.AdaptiveAvgPool2d(1))
        blocks.append(nn.Flatten())
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Linear(hidden_size[-1], num_classes)


    def f(self, x):
        rep = self.blocks(x)
        x = self.classifier(rep)
        return rep, x

    def forward(self, input):
        output = {}
        output['rep'], output['target'] = self.f(input['data'])
        if 'loss_mode' in input:
            if 'sup' in input['loss_mode']:
                output['loss'] = loss_fn(output['target'], input['target'])
            elif 'fix' in input['loss_mode'] and 'mix' not in input['loss_mode']:
                aug_rep, aug_output = self.f(input['aug'])
                output['loss'] = loss_fn(aug_output, input['target'].detach())
            elif 'fix' in input['loss_mode'] and 'mix' in input['loss_mode']:
                aug_rep, aug_output = self.f(input['aug'])
                output['loss'] = loss_fn(aug_output, input['target'].detach())
                mix_rep, mix_output = self.f(input['mix_data'])
                output['loss'] += input['lam'] * loss_fn(mix_output, input['mix_target'][:, 0].detach()) + (
                        1 - input['lam']) * loss_fn(mix_output, input['mix_target'][:, 1].detach())
            else:
                raise ValueError('Not valid loss mode')
            if cfg['task'] == 'OSR':
                if 'selected_ood_images' in input:
                    b_size_ood = min(len(input['data']),len(input['selected_ood_images']))
                    indices = torch.randperm(len(input['selected_ood_images']))[:b_size_ood]
                    # 对字典中每个可索引元素执行indices提取（保持字典结构）
                    # 1. 提取队列中指定索引的样本字典列表（原错误行修正）
                    ood_samples_list = [input['selected_ood_images'][i] for i in indices]  # deque→列表推导式

                    # 2. 将样本字典列表转换为字段对齐的张量字典（核心修复）
                    # ood_samples = {
                    #     k: torch.stack([sample[k] for sample in ood_samples_list])
                    #     for k in ood_samples_list[0].keys()  # 按第一个样本的键构建字典
                    # }
                    ood_samples = {
                        k: torch.stack([s[k].to(cfg['device']) for s in ood_samples_list if k in s])
                        for k in ['data', 'aug', 'aug1']
                        if any(k in s for s in ood_samples_list)
                    }
                    ood_scores = torch.tensor(list(input['selected_ood_scores']))[indices].to(cfg['device'])
                    ood_label = (torch.ones(b_size_ood) * (cfg['seen_class_num'])).to(cfg['device']).long()
                    all_data_ood = ood_samples['data'].to(cfg['device'])
                    rep_ood,x_ood = self.f(all_data_ood)
                    logits_ood_lb = x_ood[:b_size_ood]
                    mean_threshold = (torch.tensor(list(input['selected_ood_scores']))[indices].to(cfg['device'])).mean().item()

                    ood_mask = ood_scores < mean_threshold
                    L_sup_open = (
                            F.cross_entropy(logits_ood_lb, ood_label.to(cfg['device']), reduction='none')
                            * ood_mask
                    ).mean()
                    output['loss'] += L_sup_open
                    # L_sup_open = torch.tensor(0.0, device=cfg['device'])

        else:
            if not torch.any(input['target'] == -1):
                output['loss'] = loss_fn(output['target'], input['target'])
            if cfg['task'] == 'OSR':
                if 'prototypes' in input:
                    num_known_classes = cfg['seen_class_num']
                    threshold = input.get('ood_threshold',cfg.get('threshold_ood', cfg.get('threshold', 0))) # 获取阈值
                    # threshold = cfg.get('threshold',1)
                    proto_logits = self.classifier(input['prototypes'].to(cfg['device']))
                    # 取前 K 个 Logits (已知类部分)
                    known_logits = proto_logits[:, :num_known_classes]
                    unk_logits = proto_logits[:, -1:]

                    # 对已知类部分进行 Softmax
                    known_probs = F.softmax(known_logits, dim=1)

                    # 找出已知类中的最大概率 (MSP)
                    proto_ood_scores = torch.max(known_probs, dim=1)[0]

                    # 3. 计算 OOD 掩码
                    # 只有当 MSP 低于阈值时，原型才被认为是 OOD，需要推向开放集
                    proto_ood_mask = proto_ood_scores < threshold

                    ood_label = (torch.ones(proto_logits.shape[0]) * (num_known_classes)).to(cfg['device']).long()
                    proto_loss = (loss_fn(proto_logits,ood_label,reduction='none')*proto_ood_mask).mean()
                    output['loss'] += proto_loss
                    # print(threshold,proto_ood_mask.sum(),len(proto_ood_mask))
        return output


def wresnet28x2(momentum=None, track=False):
    data_shape = cfg['data_shape']
    if cfg['task'] == 'OSR':
        target_size = cfg['seen_class_num'] + 1
    else:
        if cfg['seen_class_num']:
            target_size = cfg['seen_class_num']
        else:
            target_size = cfg['target_size']
    depth = cfg['wresnet28x2']['depth']
    widen_factor = cfg['wresnet28x2']['widen_factor']
    drop_rate = cfg['wresnet28x2']['drop_rate']
    model = WideResNet(data_shape, target_size, depth, widen_factor, drop_rate)
    model.apply(init_param)
    model.apply(lambda m: make_batchnorm(m, momentum=momentum, track_running_stats=track))
    return model


def wresnet28x8(momentum=None, track=False):
    data_shape = cfg['data_shape']
    target_size = cfg['target_size']
    depth = cfg['wresnet28x8']['depth']
    widen_factor = cfg['wresnet28x8']['widen_factor']
    drop_rate = cfg['wresnet28x8']['drop_rate']
    model = WideResNet(data_shape, target_size, depth, widen_factor, drop_rate)
    model.apply(init_param)
    model.apply(lambda m: make_batchnorm(m, momentum=momentum, track_running_stats=track))
    return model


def wresnet37x2(momentum=None, track=False):
    data_shape = cfg['data_shape']
    target_size = cfg['target_size']
    depth = cfg['wresnet37x2']['depth']
    widen_factor = cfg['wresnet37x2']['widen_factor']
    drop_rate = cfg['wresnet37x2']['drop_rate']
    model = WideResNet(data_shape, target_size, depth, widen_factor, drop_rate)
    model.apply(init_param)
    model.apply(lambda m: make_batchnorm(m, momentum=momentum, track_running_stats=track))
    return model
