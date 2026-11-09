import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import init_param, make_batchnorm, loss_fn, mb_sup_loss_openmatch, compute_ood_proto_loss, compute_consensus, loss_mi, generate_batch_based_ood, BDMatchAttention, LayerScale, drop_path, DropPath
from config import cfg


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import init_param, make_batchnorm, loss_fn, mb_sup_loss_openmatch, compute_ood_proto_loss, compute_consensus, loss_mi, generate_batch_based_ood, BDMatchAttention, LayerScale, drop_path, DropPath, \
compute_unlabeled_loss_nedl, compute_labeled_loss_pedl
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
        self.is_1d = len(data_shape) == 2  # [Channel, Length] 为 True

        blocks = []
        hidden_size = [] # 用于存储各层特征维度，最后一位用于构建分类头

        if self.is_1d:
            # ==========================================
            # 分支 A: M5 模型结构 (针对 1D 音频优化，极速)
            # ==========================================
            input_channel = data_shape[0]
            n_channel = 32 # M5 基础通道数

            # 第一层：大卷积核模拟 STFT，大幅降采样 (16000 -> 4000)
            blocks.append(nn.Conv1d(input_channel, n_channel, kernel_size=80, stride=4))
            blocks.append(nn.BatchNorm1d(n_channel))
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(nn.MaxPool1d(4))

            # 第二层
            blocks.append(nn.Conv1d(n_channel, n_channel, kernel_size=3))
            blocks.append(nn.BatchNorm1d(n_channel))
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(nn.MaxPool1d(4))

            # 第三层
            blocks.append(nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3))
            blocks.append(nn.BatchNorm1d(2 * n_channel))
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(nn.MaxPool1d(4))

            # 第四层 (最后一层卷积)
            blocks.append(nn.Conv1d(2 * n_channel, 4 * n_channel, kernel_size=3))
            blocks.append(nn.BatchNorm1d(4 * n_channel))
            blocks.append(nn.ReLU(inplace=True))

            # 记录最终的特征维度 (M5 输出是 4*32 = 128)
            # 为了让 WideResNet 下面的代码兼容，我们构造一个 hidden_size 列表
            # 关键是 hidden_size[-1] 必须等于特征层输出的通道数
            final_feat_dim = 4 * n_channel
            hidden_size = [0, 0, 0, final_feat_dim]
            if 'bdmatch' in cfg['loss_mode']:
                # BDMatch 需要原始特征图进行 Attention，所以我们将 Pool 和 Flatten 分离
                self.blocks = nn.Sequential(*blocks) # 不包含 Pool/Flatten
                self.global_pool = nn.AdaptiveAvgPool1d(1)
                self.flatten = nn.Flatten()
            else:
                blocks.append(nn.AdaptiveAvgPool1d(1))
                blocks.append(nn.Flatten())
                self.blocks = nn.Sequential(*blocks)
        else:
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
            if 'bdmatch' in cfg['loss_mode']:
                # BDMatch 需要原始特征图进行 Attention，所以我们将 Pool 和 Flatten 分离
                self.blocks = nn.Sequential(*blocks) # 不包含 Pool/Flatten
                self.global_pool = nn.AdaptiveAvgPool2d(1)
                self.flatten = nn.Flatten()
            else:
                blocks.append(nn.AdaptiveAvgPool2d(1))
                blocks.append(nn.Flatten())
                self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Linear(hidden_size[-1], num_classes)
        if 'openmatch' in cfg['loss_mode'] or 'iomatch' in cfg['loss_mode']:
            self.ova_classifier = nn.Linear(hidden_size[-1], (cfg['seen_class_num']) * 2, bias=False)
        if 'fedopenmatch' in cfg['loss_mode']:
            self.margin = torch.zeros((2, 1), dtype=torch.float).to(cfg['device'])
            self.margin[0] = math.log(float(num_classes - 1) / float(num_classes))
            self.margin[1] = math.log(1. / float(num_classes))
        if 'divcon' in cfg['loss_mode']:
            self.num_div_heads = cfg.get('num_div_heads', 4)  # 论文推荐 K=4 或更多
            self.div_heads = nn.ModuleList([
                nn.Linear(hidden_size[-1], num_classes) for _ in range(self.num_div_heads)
            ])
            # Projector 用于互信息损失 (L_mi)
            self.projector = nn.Sequential(
                nn.Linear(hidden_size[-1], hidden_size[-1]),
                nn.ReLU(),
                nn.Linear(hidden_size[-1], 128) # 论文中 d' = 128
            )
        if 'bdmatch' in cfg['loss_mode']:
            self.num_features = hidden_size[-1]
            self.num_classes = num_classes

            # Label-Specific Attention
            self.bd_attn = BDMatchAttention(self.num_features, num_heads=4)
            self.bd_ls1 = LayerScale(self.num_features)
            self.bd_drop_path1 = DropPath(0.1) # default drop_path
            self.bd_norm = nn.LayerNorm(self.num_features, eps=1e-6)

            # Learnable Query Embedding (1, C, D)
            self.bd_query_embed = nn.Parameter(torch.zeros(1, num_classes, self.num_features))
            nn.init.normal_(self.bd_query_embed)

            # Main Branch Difference Classifier (fc2 vs original classifier)
            self.bd_fc2 = nn.Linear(self.num_features, num_classes)
            nn.init.xavier_normal_(self.bd_fc2.weight.data)
            self.bd_fc2.bias.data.zero_()

            # Standard/Auxiliary Branch Classifiers
            self.st_fc1 = nn.Linear(self.num_features, num_classes)
            self.st_fc2 = nn.Linear(self.num_features, num_classes)
            nn.init.xavier_normal_(self.st_fc1.weight.data); self.st_fc1.bias.data.zero_()
            nn.init.xavier_normal_(self.st_fc2.weight.data); self.st_fc2.bias.data.zero_()

            # Client-side: Running average of class distribution (EMA)
            # 注册为 buffer 以便保存，但不会作为参数更新
            self.register_buffer('p_model', torch.ones(self.num_classes) * 0.5)
            self.ema_decay = 0.999 # BDMatch default
            self.p_cutoff_pos = 0.95 # threshold
            self.p_cutoff_neg = 0.5  # open-set threshold

        if 'edl' in cfg['loss_mode']:
            # self.edl_classifier = nn.Sequential(
            #     nn.Linear(hidden_size[-1], hidden_size[-1]),
            #     # 将 BatchNorm1d 替换为 GroupNorm (分为 1 组等同于 LayerNorm)
            #     # GroupNorm(num_groups, num_channels)
            #     nn.GroupNorm(1, hidden_size[-1]),
            #     nn.ReLU(),
            #     nn.Linear(hidden_size[-1], hidden_size[-1]),
            #     nn.GroupNorm(1, hidden_size[-1]),
            #     nn.ReLU(),
            #     nn.Linear(hidden_size[-1], hidden_size[-1]),
            #     nn.GroupNorm(1, hidden_size[-1]),
            #     nn.ReLU(),
            #     nn.Linear(hidden_size[-1], num_classes-1, bias=False)
            # )
            self.edl_classifier = nn.Linear(hidden_size[-1], num_classes-1, bias=False)
            # 这样保证了从隐空间特征到证据空间的映射是线性的，保持了几何一致性
            self.softplus = nn.Softplus()
        else:
            print('fuck')

    def f(self, x,bd_pred=False):
        if 'bdmatch' not in cfg['loss_mode']:
            rep = self.blocks(x)
            x = self.classifier(rep)
            return rep, x
        else:
            # --- BDMatch Forward Logic ---
            # 1. 获取 Spatial Features (B, C, H, W)
            feat_map = self.blocks(x)

            # 2. 保留一份 Global Pooling 的特征用于兼容性 (output['rep'])
            rep_pool = self.global_pool(feat_map)
            rep_flat = self.flatten(rep_pool)

            # 3. 准备 Attention 输入: (B, C, H*W) -> (B, H*W, C)
            feat = feat_map.reshape((feat_map.size(0), feat_map.size(1), -1)).permute(0, 2, 1)

            # 4. 生成 Label-Specific Features
            # rep_pool (B, C, 1, 1) -> (B, 1, C) 用于残差连接
            res = rep_pool.view(-1, 1, self.num_features)

            # Apply Attention + LayerScale + DropPath
            feat_ls = res + self.bd_drop_path1(self.bd_ls1(self.bd_attn(feat, self.bd_query_embed)))
            feat_ls = self.bd_norm(feat_ls) # (B, NumClasses, Dim)

            # 5. 计算 Main Branch Logits (Difference Classifier)
            # 使用 Hadamard Product 针对每个类别计算 Logit
            # feat_ls: (B, K, D), classifier.weight: (K, D)
            logits_1 = (feat_ls * self.classifier.weight).sum(dim=-1) + self.classifier.bias
            logits_2 = (feat_ls * self.bd_fc2.weight).sum(dim=-1) + self.bd_fc2.bias
            logits = logits_1 - logits_2 # Main logits

            if not bd_pred:
                return rep_flat, logits

            # 6. 计算 Standard Branch Logits (Auxiliary)
            st_logits_1 = (feat_ls * self.st_fc1.weight).sum(dim=-1) + self.st_fc1.bias
            st_logits_2 = (feat_ls * self.st_fc2.weight).sum(dim=-1) + self.st_fc2.bias
            st_logits = st_logits_1 - st_logits_2

            return rep_flat, {'logits': logits, 'st_logits': st_logits}

    def forward(self, input):
        output = {}
        # if (input['target'] > cfg['seen_class_num']).any():
        #     print(input['target'])
        #     output['loss'] = loss_fn(cfg['seen_class_num'])
        output['rep'], output['target'] = self.f(input['data'])
        if 'openmatch' in cfg['loss_mode'] or 'iomatch' in cfg['loss_mode']:
            if 'fedopenmatch' not in cfg['loss_mode']:
                ova = self.ova_classifier(output['rep'])
            else:
                ova = self.ova_classifier(output['rep'].detach())
            output['ova'] = F.softmax(ova.view(ova.size(0), 2, -1), 1)
        if 'divcon' in cfg['loss_mode'] and torch.any(input['target'] == -1):
            # 计算所有差异头的 Logits
            # 结果形状: [Batch_Size, Num_Heads, Num_Classes]
            div_head_outputs = [head(output['rep']) for head in self.div_heads]
            output['div_head_logits'] = torch.stack(div_head_outputs, dim=1)
        if 'edl' in cfg['loss_mode']:
            output['edl_output'] = self.softplus(self.edl_classifier(output['rep'].detach())) + 1
        if 'loss_mode' in input:
            #无标签全类别客户端训练
            if 'sup' in input['loss_mode']:
                output['loss'] = loss_fn(output['target'], input['target'])
            elif 'fix' in input['loss_mode'] and 'mix' not in input['loss_mode'] and 'match' not in cfg['loss_mode'] and not any(b in input['loss_mode'] for b in cfg['baselines']):
                aug_rep, aug_output = self.f(input['aug'])
                output['loss'] = loss_fn(aug_output, input['target'].detach())
            elif 'fix' in input['loss_mode'] and 'mix' in input['loss_mode']:
                aug_rep, aug_output = self.f(input['aug'])
                if cfg['open_close_ratio']:
                    if cfg['open_close_ratio'] == 1:
                        output['loss'] = (loss_fn(aug_output[:,:-1], input['target'].detach()) + loss_fn(aug_output, input['target'].detach()))
                    else:
                        output['loss'] = loss_fn(aug_output[:,:-1], input['target'].detach()) #+ loss_fn(torch.cat([aug_output[:,:-1].detach(), aug_output[:,-1:]], dim=1), input['target'].detach()))
                        aug_rep_detached = aug_rep.detach()

                        # 步骤 B: 重新通过分类器 (只有分类器权重是变量，输入是常量)
                        aug_logits_fixed_enc = self.classifier(aug_rep_detached)

                        # 步骤 C: 拼接 Logits
                        #    - ID 部分 (detach): 作为“背景/锚点”，告诉 Loss 函数 ID 类别的分值是多少
                        #    - OOD 部分 (grad):  这是我们要压低的目标，带有针对 W_ood 的梯度
                        logits_suppression = torch.cat([
                            aug_output[:, :-1].detach(),
                            aug_logits_fixed_enc[:, -1:]
                        ], dim=1)

                        loss_suppression = loss_fn(logits_suppression, input['target'].detach())
                        output['loss'] += loss_suppression
                else:
                    output['loss'] = loss_fn(aug_output, input['target'].detach())
                if 'mix1' not in cfg['loss_mode']:
                    mix_rep, mix_output = self.f(input['mix_data'])

                    if cfg['open_close_ratio']:
                        if cfg['open_close_ratio'] == 1:
                            mix_rep_aug, mix_out_aug = self.f(input['mix_aug'])
                            ood_label = (torch.ones(len(mix_output)) * (cfg['seen_class_num'])).to(cfg['device']).long()
                            mask = input['mix_target'][:, 1] != input['mix_target'][:, 0]
                            output['loss'] += ((input['lam'] * loss_fn(mix_output[:,:-1], input['mix_target'][:, 0].detach()) + (
                                1 - input['lam']) * loss_fn(mix_output[:,:-1], input['mix_target'][:, 1].detach())) + (
                                (input['lam'] * loss_fn(mix_out_aug, input['mix_target'][:, 0].detach()) + (
                                1 - input['lam']) * loss_fn(mix_out_aug, input['mix_target'][:, 1].detach()))))# + (loss_fn(mix_out_aug, ood_label,reduction='none')*mask).mean()))/2
                        else:
                            output['loss'] += input['lam'] * loss_fn(mix_output[:,:-1], input['mix_target'][:, 0].detach()) + (
                            1 - input['lam']) * loss_fn(mix_output[:,:-1], input['mix_target'][:, 1].detach())
                    else:
                        output['loss'] += input['lam'] * loss_fn(mix_output, input['mix_target'][:, 0].detach()) + (
                            1 - input['lam']) * loss_fn(mix_output, input['mix_target'][:, 1].detach())
                if 'edl' in cfg['loss_mode'] and 'selected_id_images' in input and len(input['selected_id_images']) > 0:
                    # 1. 确定批量大小并获取随机索引 (与 OOD 逻辑完全对齐)
                    b_size_id = min(len(input['data']), len(input['selected_id_images']))
                    indices_id = torch.randperm(len(input['selected_id_images']))[:b_size_id]

                    # 2. 提取队列中指定索引的 ID 样本字典列表
                    id_samples_list = [input['selected_id_images'][i] for i in indices_id]

                    # 3. 将样本字典列表转换为字段对齐的张量字典
                    id_samples = {
                        k: torch.stack([s[k].to(cfg['device']) for s in id_samples_list if k in s])
                        for k in ['data', 'aug', 'aug1']
                        if any(k in s for s in id_samples_list)
                    }

                    # 4. 提取对应的伪标签 (从之前筛选好的 selected_id_labels 队列中取)
                    id_labels = torch.tensor(list(input['selected_id_labels']))[indices_id].to(cfg['device']).long()

                    # 5. 确定送入网络的数据 (EDL 通常使用原图 'data')
                    if cfg['ood_data'] == 'aug':
                        all_data_id = id_samples['data'].to(cfg['device'])
                    else:
                        all_data_id = id_samples['data'].to(cfg['device'])

                    # 6. 前向传播提取特征
                    rep_id, x_id = self.f(all_data_id)

                    # 7. 计算 EDL 证据 (Evidence) 和 狄利克雷参数 Alpha
                    # 【注意】：由于这是 ID 样本，我们通常希望它的梯度能够回传并微调骨干网络(backbone)
                    # 所以这里不像 OOD 那样使用 rep_ood.detach()，而是直接使用 rep_id
                    id_edl_evidence = self.softplus(self.edl_classifier(rep_id.detach()))
                    id_alpha = id_edl_evidence + 1.0

                    # 8. 获取超参数并计算 P-EDL 损失
                    f_c = cfg.get('fisher_c', 0.01)
                    epoch = cfg.get('global_epoch', 0)

                    # L_pedl_id = compute_labeled_loss_pedl(id_alpha,id_labels,cfg['global_epoch']*100,know_id=False, fisher_c=f_c)
                    if cfg.get('pedl_client',1):
                        L_pedl_id = compute_labeled_loss_pedl(id_alpha,id_labels,cfg['global_epoch']*100,know_id=False, fisher_c=f_c)
                    else:
                        L_pedl_id = torch.zeros(1, device=cfg['device']).squeeze()

                    # 9. 累加到总损失中
                    # 由于这是客户端基于伪标签的训练，可以引入一个安全的权重比例防崩（如果没有配，默认 1.0）
                    pedl_weight = cfg.get('pedl_weight', 1.0)
                    output['loss'] += L_pedl_id * pedl_weight
                    # print(L_pedl_id)
            elif 'openmatch' in cfg['loss_mode'] and 'fedopenmatch' not in cfg['loss_mode']:
                aug_rep, aug_output = self.f(input['aug'])
                logits_ova1 = ova
                logits_ova2 = self.ova_classifier(aug_rep)
                # entropy minimization for ova_logits
                logits_ova1, logits_ova2 = F.softmax(logits_ova1.view(logits_ova1.size(0), 2, -1), 1), F.softmax(logits_ova2.view(logits_ova2.size(0), 2, -1), 1)
                ls_oem = torch.mean(torch.sum(-logits_ova1 * torch.log(logits_ova1 + 1e-8), 1)) + torch.mean(torch.sum(-logits_ova2 * torch.log(logits_ova2 + 1e-8), 1))
                # consistency regularization
                ls_ova_cr = torch.mean(torch.sum(torch.sum((logits_ova1 - logits_ova2)**2, 1), 1))
                loss = ls_ova_cr * 0.5 + ls_oem * 0.5 * 0.1
                max_probs, pl = torch.max(torch.softmax(output['target'],dim=-1).detach(), dim=-1)
                fix_mask = max_probs.ge(cfg.get('threshold',0.95))
                loss += (F.cross_entropy(aug_output, pl, reduction='none') * fix_mask).mean()
                output['loss'] = loss
                output['ova'] = logits_ova1
            elif 'iomatch' in cfg['loss_mode']:
                with torch.no_grad():
                    # generator closed-set and open-set targets (pseudo-labels)
                    logits_w = output['target']
                    logits_ova = ova
                    pl_close = F.softmax(logits_w, dim=-1)
                    num_ulb = input['target'].size(0)
                    logits_ova = logits_ova.view(num_ulb, 2, -1)
                    ova = F.softmax(logits_ova, 1)
                    output['ova'] = ova
                    tmp_range = torch.arange(0, num_ulb).long().to(cfg['device'])
                    o_neg = ova[tmp_range, 0, :]
                    o_pos = ova[tmp_range, 1, :]
                    pl_open = torch.zeros((num_ulb, cfg['seen_class_num'] + 1)).to(cfg['device'])
                    pl_open[:, :cfg['seen_class_num']] = pl_close[:,:cfg['seen_class_num']] * o_pos                 # inlier score
                    pl_open[:, cfg['seen_class_num']] = torch.sum(pl_close[:,:cfg['seen_class_num']] * o_neg, 1)    # outlier score
                    out_scores = torch.sum(pl_close[:,:cfg['seen_class_num']] * o_neg, 1)   # outlier score
                    in_mask = (out_scores < 0.5)
                    close_conf, close_lb = torch.max(pl_close[:,:cfg['seen_class_num']], 1)
                    open_conf, open_lb = torch.max(pl_open, 1)
                    close_mask = close_conf.ge(cfg.get('threshold',0.95))
                    open_mask = open_conf.ge(cfg.get('open_cutoff',0.5))

                aug_rep, aug_output = self.f(input['aug'])
                logits_s = aug_output
                logits_open_s = self.ova_classifier(aug_rep)
                loss = torch.tensor(0., requires_grad=True).to(cfg['device'])
                if any(in_mask * close_mask):
                    mask = in_mask * close_mask
                    ui_loss = (F.cross_entropy(logits_s, close_lb, reduction='none') * mask.float()).mean()
                    loss += ui_loss
                if any(open_mask):
                    op_loss = (F.cross_entropy(logits_open_s, open_lb, reduction='none') * open_mask.float()).mean()
                    loss += op_loss
                output['loss'] = loss
            elif 'fedopenmatch' in cfg['loss_mode']:
                aug_rep, aug_output = self.f(input['aug'])
                logits_ova1 = ova
                logits_ova2 = self.ova_classifier(aug_rep)
                logits_w, logits_ova_w = output['target'].detach(), logits_ova1.detach()
                logits_s = aug_output
                logits_ova_s = logits_ova2
                logits_ova_pos = logits_ova_s.view(logits_ova_s.size(0), 2, -1).softmax(1)[:, 1, :]

                cr_ls =  F.mse_loss(logits_w, logits_s) * 0.5 + F.mse_loss(logits_ova_w, logits_ova_s) * 0.5
                loss = cr_ls
                if True:
                    loss += F.cross_entropy(logits_s, input['target'], reduction='mean')
                if True:
                    ova_pl = F.one_hot(input['target'].long(), num_classes=cfg['seen_class_num']).float()
                    loss += F.binary_cross_entropy(logits_ova_pos, ova_pl.float(), reduction='mean') * cfg['seen_class_num']
                output['loss'] = loss
            elif 'divcon' in cfg['loss_mode']:
                strong_data = input['aug']

                # 提取强增强特征
                rep_s, logits_s = self.f(strong_data)

                # 特征复用：rep_w 已经通过 self.f(input['data']) 算出，即 output['rep']
                rep_w = output['rep']
                logits_w = output['target']

                # --- Step A: Diversify (训练差异头) ---
                probs_w_list = [] # 收集所有头的弱增强预测概率
                loss_ssl_heads = 0.0

                for head in self.div_heads:
                    # 1. 实时生成该头的伪标签 (使用弱增强)
                    # .detach() 截断梯度，确保只更新 Head，不影响 Encoder
                    head_logits_w = head(rep_w.detach())
                    head_prob_w = torch.softmax(head_logits_w, dim=-1)
                    probs_w_list.append(head_prob_w) # 存入列表用于计算 Consensus

                    # 2. 获取伪标签
                    max_probs, pseudo_label = torch.max(head_prob_w, dim=-1)
                    mask = max_probs.ge(cfg.get('threshold', 0.95))

                    # 3. 计算该头在强增强数据上的预测
                    head_logits_s = head(rep_s.detach())

                    # 4. 计算该头的 FixMatch Loss
                    loss_ssl_heads += (F.cross_entropy(head_logits_s, pseudo_label, reduction='none') * mask).mean()
                output['div_head_logits'] = torch.stack(probs_w_list, dim=1)

                # --- Step B: Calculate Mutual Information Loss (L_mi) ---
                # 在无标签数据上计算，迫使各头产生分歧
                # 调用 utils.py 中的 loss_mi
                l_div_mi = loss_mi(probs_w_list)
                lambda_mi = cfg.get('lambda_mi', 0.5)

                # 差异头总 Loss
                L_div = loss_ssl_heads + lambda_mi * l_div_mi

                # --- Step C: Conquer (计算共识度与权重) ---
                # 调用 utils.py 中的 compute_consensus
                s_raw = compute_consensus(probs_w_list)

                # 归一化 (Batch-wise approximation)
                s_min, s_max = s_raw.min(), s_raw.max()
                if s_max - s_min > 1e-6:
                    s_norm = (s_raw - s_min) / (s_max - s_min)
                else:
                    s_norm = s_raw

                output['consensus'] = s_norm # 保存以备查

                # --- Step D: Soft Rejection (加权) ---
                tau_open = cfg.get('tau_open', 0.5) # 阈值 (Otsu阈值近似)
                t_omega = cfg.get('t_omega', 1.5)   # 温度系数

                weights = torch.ones_like(s_norm)
                outlier_mask = s_norm < tau_open
                # 软拒绝公式: (s / tau)^t
                weights[outlier_mask] = (s_norm[outlier_mask] / tau_open) ** t_omega

                # --- Step E: Main Classifier Loss ---
                # 主头 FixMatch，但加上了 weights (Consensus Weighting)
                probs_main = torch.softmax(logits_w, dim=-1)
                max_probs_main, pl_main = torch.max(probs_main, dim=-1)
                mask_main = max_probs_main.ge(cfg.get('threshold', 0.95))

                # 如果是 Outlier，weights 会很小，从而在 Loss 中被忽略
                L_u = (F.cross_entropy(logits_s, pl_main, reduction='none') * mask_main * weights).mean()

                output['loss'] = L_u + L_div
            elif 'bdmatch' in cfg['loss_mode']:
                # 获取弱增强和强增强
                x_w = input['data']
                x_s = input['aug']

                # --- Step 1: Weak Augmentation 推理 ---
                _, outs_w = self.f(x_w, bd_pred=True)
                logits_w = outs_w['logits']
                st_logits_w = outs_w['st_logits']

                prob_w = logits_w.sigmoid() # BDMatch 使用 Sigmoid (二元分解)

                # --- Step 2: Adaptive Logit Adjustment (更新 p_model) ---
                if self.training:
                    with torch.no_grad():
                        # Update distribution estimation
                        self.p_model = self.p_model * self.ema_decay + (1 - self.ema_decay) * prob_w.mean(dim=0)

                        # Calculate Adjustment term
                        p = self.p_model
                        logits_adj = torch.log(p + 1e-8) - torch.log(1.0 - p + 1e-8)

                        # Apply adjustment (tau usually 1.0)
                        prob_adj = (logits_w - 1.0 * logits_adj).sigmoid()

                        # --- Step 3: Generate Pseudo-Labels (Binary) ---
                        # Main Branch Pseudo-Labels
                        # cutoff_pos (0.95), cutoff_neg (0.5 or lower)
                        max_probs, pl = torch.max(prob_adj, dim=-1) # 这只是用来找最大类，实际用 mask 判断

                        # 正样本 Mask (确信属于该类)
                        pos_mask = prob_adj.ge(self.p_cutoff_pos)
                        # 负样本 Mask (确信不属于该类 - 包含 OOD)
                        neg_mask = prob_adj.le(self.p_cutoff_neg)

                        # 结合 Mask
                        mask = (pos_mask | neg_mask).float()

                        # 生成二元伪标签 (1 if pos, 0 if neg)
                        # 注意：这里的伪标签是针对每个类别独立的 (B, C)
                        pseudo_label = pos_mask.float()

                        # Standard Branch Pseudo-Labels (不带 Logit Adjustment)
                        prob_st = st_logits_w.sigmoid()
                        st_pos_mask = prob_st.ge(self.p_cutoff_pos)
                        st_neg_mask = prob_st.le(self.p_cutoff_neg)
                        st_mask = (st_pos_mask | st_neg_mask).float()
                        st_pseudo_label = st_pos_mask.float()

                # --- Step 4: Strong Augmentation 训练 ---
                _, outs_s = self.f(x_s, bd_pred=True)
                logits_s = outs_s['logits']
                st_logits_s = outs_s['st_logits']

                # --- Step 5: Loss Calculation (BCE) ---
                # 使用 BCEWithLogitsLoss
                bce = nn.BCEWithLogitsLoss(reduction='none')

                # Main Branch Loss
                loss_main = (bce(logits_s, pseudo_label) * mask).mean() * self.num_classes

                # Standard Branch Loss
                loss_st = (bce(st_logits_s, st_pseudo_label) * st_mask).mean() * self.num_classes

                output['loss'] = loss_main + loss_st
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
                    if cfg['ood_data'] == 'aug':
                        if 'edl' in cfg['loss_mode']:
                            all_data_ood = ood_samples['data'].to(cfg['device'])
                        else:
                            all_data_ood = ood_samples['aug'].to(cfg['device'])
                    else:
                        all_data_ood = ood_samples['data'].to(cfg['device'])
                    rep_ood,x_ood = self.f(all_data_ood)
                    if 'edl' not in cfg['loss_mode']:
                        logits_ood_lb = x_ood[:b_size_ood]
                        logits_ood_lb = torch.cat([logits_ood_lb[:,:-1].detach(), logits_ood_lb[:,-1:]], dim=1)
                        if cfg['ood_thre']:
                            mean_threshold = cfg['ood_thre']
                        else:
                            mean_threshold = (torch.tensor(list(input['selected_ood_scores']))[indices].to(cfg['device'])).mean().item()


                        ood_mask = ood_scores < mean_threshold
                        L_sup_open = (
                                F.cross_entropy(logits_ood_lb, ood_label.to(cfg['device']), reduction='none')
                                * ood_mask
                        ).mean()
                        if 'weight' in cfg and cfg['weight']:
                            weight = cfg['weight']
                        else:
                            weight = (cfg['target_size'] - cfg['seen_class_num']) / cfg['target_size']
                        # if cfg['data_name'] == 'CIFAR10':
                        # if 'CIFAR10' in cfg['data_name']:
                        #     weight = 1/cfg['seen_class_num']
                        output['loss'] += L_sup_open*weight
                        # L_sup_open = torch.tensor(0.0, device=cfg['device'])
                    if 'edl' in cfg['loss_mode'] and cfg['global_epoch'] >=10:
                        # 1. 获取 OOD 样本的特征表示（基于前面计算好的 rep_ood）
                        # 注意：需要确保 self.edl_classifier 接收的是特征向量
                        # 如果你的 edl_classifier 包含在 self.f 后面，可以直接调用
                        ood_edl_evidence = self.softplus(self.edl_classifier(rep_ood.detach()))

                        # 2. 将证据转换为 Dirichlet 分布的参数 alpha (alpha = evidence + 1)
                        # 这是 EDL 的标准做法，确保证据为非负
                        ood_alpha = ood_edl_evidence + 1.0

                        # 3. 调用你写好的负向证据损失函数
                        # 这里的 fisher_c 可以从 cfg 中读取，如果没有则默认为 0.01
                        f_c = cfg.get('fisher_c', 0.01)
                        # L_nedl_ood = compute_unlabeled_loss_nedl(ood_alpha, fisher_c=f_c,know_ood=True)
                        if cfg.get('nedl_client',1):
                            L_nedl_ood = compute_unlabeled_loss_nedl(ood_alpha, fisher_c=f_c,know_ood=True)
                        else:
                            L_nedl_ood = torch.zeros(1, device=cfg['device']).squeeze()
                        output['loss'] += L_nedl_ood


        else:
            #有标签id服务器端训练
            if not torch.any(input['target'] == -1):
                if ('openmatch' in cfg['loss_mode'] or 'iomatch' in cfg['loss_mode']) and 'fedopenmatch' not in cfg['loss_mode']:
                    logits = output['target']
                    logits_ova = ova
                    ce_loss = loss_fn(logits, input['target'])
                    # open-set detection
                    ova_loss = mb_sup_loss_openmatch(logits_ova, input['target'])
                    loss = ce_loss + ova_loss
                    output['loss'] = loss
                elif 'fedopenmatch' in cfg['loss_mode']:
                    logits = output['target']
                    logits_ova = ova.view(logits.size(0), 2, -1)
                    sup_closed_loss = loss_fn(logits, input['target'])
                    logits_ova += self.margin
                    sup_mb_loss = mb_sup_loss_openmatch(logits_ova, input['target'])
                    output['loss'] = sup_closed_loss + sup_mb_loss
                elif 'divcon' in cfg['loss_mode']:
                    loss_main = loss_fn(output['target'], input['target'])

                    # 2. 差异头 Loss (Supervised)
                    # 重要：所有差异头在有标签数据上必须“达成共识”，即都预测正确标签
                    loss_div_sup = 0.0
                    rep_sup = output['rep'] # 复用特征
                    div_head_outputs = []

                    for head in self.div_heads:
                        pred = head(rep_sup)
                        loss_div_sup += loss_fn(pred, input['target'])
                        div_head_outputs.append(pred)
                    output['div_head_logits'] = torch.stack(div_head_outputs, dim=1)

                    # 服务端不需要计算 L_mi，因为有标签数据上我们希望它们一致正确，而不是通过差异化来探索。
                    # 差异化只在无标签数据(Client端)上通过 L_mi 强制产生。

                    output['loss'] = loss_main + loss_div_sup
                elif 'bdmatch' in cfg['loss_mode']:
                    # 有标签数据是 Closed-Set，直接训练
                    # 将 targets 转换为 One-Hot (B, C)
                    targets_one_hot = F.one_hot(input['target'].long(), num_classes=self.num_classes).float()

                    _, outs = self.f(input['data'], bd_pred=True)
                    logits = outs['logits']
                    st_logits = outs['st_logits']

                    # Margin for supervised learning (Paper eq 6 approx)
                    # BDMatch uses a margin to suppress negative classes further
                    # sup_remargin = - tau * log(C-1)
                    tau = 1.0
                    sup_remargin = - tau * math.log(self.num_classes - 1)

                    bce = nn.BCEWithLogitsLoss(reduction='none')

                    # Main Branch Supervised Loss (add margin to logits)
                    loss_sup = bce(logits + sup_remargin, targets_one_hot).mean() * self.num_classes

                    # Standard Branch Supervised Loss
                    loss_st_sup = bce(st_logits, targets_one_hot).mean() * self.num_classes

                    output['loss'] = loss_sup + loss_st_sup
                elif cfg['open_close_ratio'] and cfg['task'] == 'OSR':
                    # output['loss'] = (loss_fn(output['target'][:,:-1], input['target']) + loss_fn(output['target'], input['target']))/2
                    if cfg['open_close_ratio'] == 1:
                        output['loss'] = (loss_fn(output['target'][:,:-1], input['target'].detach()) + loss_fn(output['target'], input['target'].detach()))
                    else:
                        output['loss'] = loss_fn(output['target'][:,:-1], input['target'].detach())# + loss_fn(torch.cat([output['target'][:,:-1].detach(), output['target'][:,-1:]], dim=1), input['target'].detach()))
                        rep_detached = output['rep'].detach()
                        logits_fixed_enc = self.classifier(rep_detached)
                        logits_for_suppression = torch.cat([
                            output['target'][:, :-1].detach(),
                            logits_fixed_enc[:, -1:]
                        ], dim=1)
                        loss_suppression = loss_fn(logits_for_suppression, input['target'].detach())
                        output['loss'] += loss_suppression
                    if 'unlabeled' in input:
                        # 1. 获取无标签数据的 Logits
                        # input['unlabeled']['data'] 是弱增强样本 (用于生成伪标签)
                        # input['unlabeled']['aug']  是强增强样本 (用于计算损失)
                        _, unl_logits_all = self.f(input['unlabeled']['data'])      # 弱增强输出
                        _, unl_aug_logits_all = self.f(input['unlabeled']['aug'])   # 强增强输出

                        # 2. 限制在已知类范围内 (前 seen_class_num 个类)
                        K = cfg['seen_class_num']
                        unl_logits_weak = unl_logits_all[:, :K]      # 弱增强的前 K 类
                        unl_logits_strong = unl_aug_logits_all[:, :K] # 强增强的前 K 类

                        # 3. 基于弱增强样本生成伪标签和置信度 Mask
                        with torch.no_grad():
                            # 在已知类维度上做 Softmax 得到分布
                            unl_probs = torch.softmax(unl_logits_weak, dim=1)
                            max_probs, pseudo_labels = torch.max(unl_probs, dim=1)

                            # 0.95 置信度阈值筛选
                            threshold = 0.95
                            mask = max_probs.ge(threshold).float()

                        # 4. 在强增强样本上计算损失
                        # 目标是弱增强生成的 pseudo_labels，预测是强增强的 unl_logits_strong
                        unl_loss = (F.cross_entropy(unl_logits_strong, pseudo_labels, reduction='none') * mask).mean()

                        # 5. 合并损失
                        ssl_weight = 0.2 # 你设置的权重
                        if 'loss' not in output:
                            output['loss'] = torch.tensor(0.0).to(unl_logits_strong.device)

                        # 累加到总损失中
                        output['loss'] += ssl_weight * unl_loss

                        # 记录中间变量（可选，用于监控）
                        output['unl_mask_ratio'] = mask.mean() # 监控有多少样本通过了 0.95 阈值

                else:
                    output['loss'] = loss_fn(output['target'], input['target'])
                    if 'unlabeled' in input:
                        # 1. 获取无标签数据的 Logits
                        # input['unlabeled']['data'] 是弱增强样本 (用于生成伪标签)
                        # input['unlabeled']['aug']  是强增强样本 (用于计算损失)
                        _, unl_logits_all = self.f(input['unlabeled']['data'])      # 弱增强输出
                        _, unl_aug_logits_all = self.f(input['unlabeled']['aug'])   # 强增强输出
                        unl_logits_weak = unl_logits_all
                        unl_logits_strong = unl_aug_logits_all
                        # 3. 基于弱增强样本生成伪标签和置信度 Mask
                        with torch.no_grad():
                            # 在已知类维度上做 Softmax 得到分布
                            unl_probs = torch.softmax(unl_logits_weak, dim=1)
                            max_probs, pseudo_labels = torch.max(unl_probs, dim=1)
                            # 0.95 置信度阈值筛选
                            threshold = 0.95
                            mask = max_probs.ge(threshold).float()
                        # 4. 在强增强样本上计算损失
                        # 目标是弱增强生成的 pseudo_labels，预测是强增强的 unl_logits_strong
                        unl_loss = (F.cross_entropy(unl_logits_strong, pseudo_labels, reduction='none') * mask).mean()
                        # 5. 合并损失
                        ssl_weight = 0.2 # 你设置的权重
                        if 'loss' not in output:
                            output['loss'] = torch.tensor(0.0).to(unl_logits_strong.device)
                        # 累加到总损失中
                        output['loss'] += ssl_weight * unl_loss
                        # 记录中间变量（可选，用于监控）
                        output['unl_mask_ratio'] = mask.mean() # 监控有多少样本通过了 0.95 阈值
                if 'edl' in cfg['loss_mode']:
                    f_c = cfg.get('fisher_c',0.01)
                    loss_pedl = compute_labeled_loss_pedl(output['edl_output'],input['target'].detach(),cfg['global_epoch']*100,fisher_c=f_c)
                    alp = output['edl_output']
                    alp0 = torch.sum(alp, dim=-1, keepdim=True)
                    num_classes = alp.size(1)

                    # 1. 负向 MSE 目标：均匀分布
                    labels_uniform = torch.ones_like(alp) / num_classes
                    gamma1_alp = torch.polygamma(1, alp)
                    gamma1_alp0 = torch.polygamma(1, alp0)


                    # 2. Fisher 信息项：几何约束压制
                    # loss_det_fisher = - (torch.log(gamma1_alp).sum(-1) +
                    #                      torch.log(1.0 - (gamma1_alp0 / gamma1_alp).sum(-1))).mean()
                    eps = 1e-8
                    term_fisher_inner = 1.0 - (gamma1_alp0 / (gamma1_alp + eps)).sum(-1)

                    output['loss'] += loss_pedl

            if cfg['task'] == 'OSR':
                if 'prototypes' in input and 'promax1' not in cfg['sel_method'] and 'subspace' not in cfg['sel_method']:
                    # print('shit')
                    # import sys
                    # sys.exit()
                    num_known_classes = cfg['seen_class_num']
                    if cfg['ood_thre']:
                        threshold = cfg['ood_thre']
                    else:
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
                    proto_loss,_ = compute_ood_proto_loss(output['rep'].detach(), input['prototypes'].to(cfg['device']), proto_logits, num_known_classes, cfg['device'])
                    # proto_loss = (loss_fn(proto_logits,ood_label,reduction='none')*proto_ood_mask).mean()
                    print(output['loss'],proto_loss)
                    output['loss'] += proto_loss


                current_valid = input.get('num_valid_classes', 0)
                seen_num = cfg['seen_class_num']

                # 只有当统计量覆盖了所有类别时才开始，防止生成偏置
                if 'all_means' in input and 'valid_class_mask' in input and 'subspaces' in cfg['sel_method'] and 'affinity' not in cfg['sel_method']:


                    seen_num = cfg['seen_class_num']
                    targets = input['target']

                    # 1. 获取 Mask 并转到当前设备
                    # server_mask: [C] (Boolean)
                    server_mask = input['valid_class_mask'].to(cfg['device'])

                    # 2. 生成有效样本掩码 (Valid Sample Mask)
                    # 步骤 A: 必须是 ID 类别 (防止非法索引)
                    is_id_sample = targets < seen_num

                    # 步骤 B: 该样本所属的类别，必须在 Server 端已有统计量
                    # 我们先初始化一个全 False 的 mask
                    has_stats_sample = torch.zeros_like(is_id_sample)

                    # 仅对 ID 样本检查 Server Mask (利用高级索引)
                    # 逻辑：查找 targets 对应的类别在 server_mask 中是否为 True
                    if is_id_sample.any():
                        has_stats_sample[is_id_sample] = server_mask[targets[is_id_sample]]

                    # 最终掩码：是 ID 且 有统计量
                    valid_batch_mask = is_id_sample & has_stats_sample

                    # 3. 如果当前 Batch 有符合条件的样本，开始生成
                    if valid_batch_mask.any():
                        # --- 准备数据 ---
                        valid_targets = targets[valid_batch_mask]
                        valid_reps = output['rep'][valid_batch_mask].detach() # [B_valid, D]

                        # # 准备 Client 聚合来的统计量
                        # agg_means = input['all_means'].to(cfg['device'])
                        # agg_vars = input['all_vars'].to(cfg['device'])

                        # # --- [方差校正] 计算当前 Batch 的实时方差 ---
                        # batch_vars_per_class = torch.zeros_like(agg_vars)
                        # unique_c = torch.unique(valid_targets)

                        # for c in unique_c:
                        #     mask_c = (valid_targets == c)
                        #     if mask_c.sum() > 1:
                        #         batch_vars_per_class[c] = torch.var(valid_reps[mask_c], dim=0, unbiased=True)

                        # # 取最大值 (Server聚合 vs 当前Batch实时)
                        # effective_vars = torch.max(agg_vars, batch_vars_per_class)
                        # # 膨胀 1.1 倍
                        # effective_stds = torch.sqrt(effective_vars + 1e-6) * 1.1

                        # # --- [核心生成] ---
                        # # 索引对应的均值和方差
                        # batch_means = agg_means[valid_targets]
                        # batch_stds = effective_stds[valid_targets]
                        # B_gen, D_gen = batch_means.shape

                        # # === 修改开始: 动态边界 k_min 与高斯尾部采样 ===

                        # # 1. 计算当前样本相对于类中心的标准化距离 (Sigma Distance)
                        # #    z_score: [B_gen, D]
                        # z_score = (valid_reps - batch_means) / (batch_stds + 1e-8)
                        # #    dists: [B_gen, 1]
                        # dists = torch.norm(z_score, p=2, dim=1, keepdim=True)

                        # # 2. 动态 k_min: 设定每个类别的生成下界为当前Batch中该类的最大距离
                        # #    这样保证 OOD 样本刚好生成在 ID 样本的边界之外
                        # k_min = torch.zeros_like(dists)
                        # unique_c = torch.unique(valid_targets)
                        # for c in unique_c:
                        #     mask_c = (valid_targets == c)
                        #     if mask_c.any():
                        #         k_min[mask_c] = dists[mask_c].max()

                        # # 3. 动态 scales: 采样自截断高斯分布 (Truncated Normal)
                        # #    范围 [k_min, +∞)，符合高斯分布特性 (离中心越远，概率密度越低)
                        # #    使用 Inverse Transform Sampling 实现
                        # sqrt2 = 1.41421356

                        # # 计算截断点 k_min 处的高斯累积概率 CDF
                        # cdf_min = 0.5 * (1 + torch.erf(k_min / sqrt2))

                        # # 在有效概率区间 [cdf_min, 1) 内均匀采样
                        # # clamp 防止随机到 1.0 导致 erfinv 输出无穷大
                        # p_uniform = cdf_min + torch.rand(B_gen, 1, device=cfg['device']) * (1 - cdf_min)
                        # p_uniform = torch.clamp(p_uniform, max=1.0 - 1e-7)

                        # # 逆误差函数变换得到对应的 sigma 距离
                        # scales = sqrt2 * torch.erfinv(2 * p_uniform - 1)

                        # # === 修改结束 ===

                        # # Annulus Sampling (环形采样)
                        # noise = torch.randn(B_gen, D_gen, device=cfg['device'])
                        # noise_norm = torch.norm(noise, p=2, dim=1, keepdim=True) + 1e-8
                        # normalized_noise = noise / noise_norm

                        # # (原代码的 scales 计算部分已被替换)

                        # # 合成 OOD 特征
                        # generated_ood_reps = batch_means + scales * batch_stds * normalized_noise

                        # === [步骤 1] 构建基于"超长方体"的向量边界 ===

                        unique_c = torch.unique(valid_targets)

                        pure_centers_map = {}   # 存储均值向量 [D]
                        boundary_vec_map = {}   # 存储边界向量 [D] (每个维度上的最大偏差)

                        for c in unique_c:
                            mask_c = (valid_targets == c)
                            reps_c = valid_reps[mask_c] # [N_c, D]

                            # 1. 计算 ID 中心 (Vector)
                            center_c = reps_c.mean(dim=0)
                            pure_centers_map[c.item()] = center_c

                            # 2. 计算每个维度上的最大绝对偏差 (Boundary Vector)
                            # abs_diff: [N_c, D]
                            abs_diff = torch.abs(reps_c - center_c)

                            # max_dev: [D] (这是一个向量！代表每个方向上的最远距离)
                            # 如果样本只有一个，max_dev 就是 0，这会导致问题，所以加一个极小值 eps
                            max_dev_c = torch.max(abs_diff, dim=0)[0] + 1e-6

                            boundary_vec_map[c.item()] = max_dev_c

                        # B. 对齐到 Tensor
                        # [B_gen, D]
                        batch_pure_centers = torch.stack([pure_centers_map[t.item()] for t in valid_targets])
                        # [B_gen, D] <--- 注意这里是 D 维向量，不再是 [B, 1]
                        batch_boundary_vecs = torch.stack([boundary_vec_map[t.item()] for t in valid_targets])

                        # === [步骤 2] 从混合分布生成 (All-Means/Vars) ===

                        # [B_gen, D]
                        source_mean = input['all_means'][valid_targets]
                        # [B_gen, D]
                        source_std = torch.sqrt(input['all_vars'][valid_targets] + 1e-6)

                        B_gen, D_gen = valid_reps.shape
                        noise = torch.randn(B_gen, D_gen, device=cfg['device'])
                        raw_ood_reps = source_mean + noise * source_std

                        # === [步骤 3] 逐维度投影修正 (Element-wise Projection) ===

                        # 1. 计算相对于 ID 中心的原始偏移向量
                        # [B_gen, D]
                        diff_vec = raw_ood_reps - batch_pure_centers

                        # 2. 计算绝对偏移量
                        abs_diff = torch.abs(diff_vec)

                        # 3. 核心逻辑：逐维度比较与修正
                        # 你的要求："在每个方向上都偏移 ID 分布"
                        # 意味着：对于维度 i，如果 abs_diff[i] < boundary[i]，则强制设为 boundary[i]
                        # 如果 abs_diff[i] 已经很大（在外部），则保留原值

                        # target_abs_diff: [B_gen, D]
                        # max(生成的偏差, ID的最大偏差)
                        # 这保证了 OOD 样本在每一个维度上，都至少和 ID 样本中最远的那个一样远
                        target_abs_diff = torch.max(abs_diff, batch_boundary_vecs)

                        # 稍微加一点点 epsilon 膨胀 (可选)，确保严格在边界外
                        target_abs_diff = target_abs_diff + 1e-6

                        # 4. 恢复符号并合成
                        # sign(diff) 保持了原始生成的方向（左边还是右边）
                        generated_ood_reps = batch_pure_centers + torch.sign(diff_vec) * target_abs_diff

                        # --- [计算 Loss] ---
                        ood_logits = self.classifier(generated_ood_reps)
                        ood_targets = torch.full((B_gen,), seen_num, device=cfg['device'], dtype=torch.long)
                        id_logits_part = ood_logits[:, :seen_num] # [B, K]
                        ood_logit_part = ood_logits[:, seen_num:] # [B, 1]


                        # === [你的方案] Logit Aggregation ===

                        # 1. 计算 ID 类的平均 Logit
                        # keepdim=True 保持形状为 [B, 1]
                        if cfg['binary_method']:
                            if cfg['binary_method'] == 'mean':
                                id_avg_logit = torch.mean(id_logits_part, dim=1, keepdim=True)
                            elif cfg['binary_method'] == 'max':
                                id_avg_logit = torch.max(id_logits_part, dim=1, keepdim=True)
                            elif cfg['binary_method'] == 'lse':
                                # --- 核心修改 ---
                                # T 控制梯度的平滑程度。
                                # CIFAR-10 可以用 T=1.0 (接近 Max，保证概率高)
                                # CIFAR-100 可以用 T=2.0 ~ 5.0 (接近 Mean，防止压坏尖锐特征)
                                # correction = T * torch.log(torch.tensor(100.0))

                                T = cfg.get('energy_temp', 1.0)
                                if cfg['data_name'] in ['CIFAR10','SVHN']:
                                    T = 1
                                elif cfg['data_name'] in ['CIFAR100']:
                                    T = 10
                                correction = T * torch.log(torch.tensor(float(seen_num), device=cfg['device']))

                                id_avg_logit = T * torch.logsumexp(id_logits_part / T, dim=1, keepdim=True) - correction
                            elif cfg['binary_method'] == 'shifted_mean':
                                # --- 终极平衡方案：数值取 Max，梯度取 Mean ---

                                # 1. 计算均值 (Mean)
                                # 梯度的提供者：Mean 对所有 logit 的导数都是 1/N。
                                # 如果 Loss 降低 id_avg，等同于把所有 id_logits 减去同一个值。
                                # 数学性质：Variance (尖锐度) 不变！
                                id_avg = torch.mean(id_logits_part, dim=1, keepdim=True)

                                # 2. 计算最大值 (Max) 与均值的差值 (Gap)
                                # 我们只关心这个“尖锐度 Gap”的大小，不需要对它求导。
                                # 用 .detach() 把它固定住。
                                id_max_val, _ = torch.max(id_logits_part, dim=1, keepdim=True)
                                sharpness_gap = (id_max_val - id_avg).detach()

                                # 3. 构造“虚拟 ID 分数”
                                # Forward 时: id_virtual = id_avg + (id_max - id_avg) = id_max
                                # Backward 时: grad(id_virtual) = grad(id_avg) + 0 = 均匀梯度
                                id_avg_logit = id_avg + sharpness_gap

                            binary_logits = torch.cat([id_avg_logit, ood_logit_part], dim=1)
                            binary_targets = torch.ones(B_gen, device=cfg['device'], dtype=torch.long)

                            # loss_ood = F.cross_entropy(ood_logits, ood_targets)
                            loss_ood = F.cross_entropy(binary_logits, binary_targets)

                        else:
                            loss_ood = F.cross_entropy(ood_logits, ood_targets)

                        # 合并 Loss
                        # print(output['loss'].item(),loss_ood.item())
                        if 'weight' in cfg and cfg['weight']:
                            weight = cfg['weight']
                        else:
                            weight = (cfg['target_size'] - cfg['seen_class_num']) / cfg['target_size']
                        # if cfg['data_name'] == 'CIFAR10':
                        # if 'CIFAR10' in cfg['data_name']:
                        #     weight = 1/cfg['seen_class_num']
                        output['loss'] += loss_ood * weight

                if 'affinity' in cfg['sel_method'] and 'batch_affinity' in input:

                    # 1. 获取当前 Batch 的真实特征和标签
                    curr_reps = output['rep']    # [B, D]
                    curr_targets = input['target'] # [B]
                    if 'randommixserver' in cfg['loss_mode']:
                        # 为了不影响主干网络的梯度更新，通常建议 detach
                        detached_reps = curr_reps.detach()

                        # 计算当前 Batch 在特征维度上的均值和标准差
                        # mean: [D], std: [D]
                        batch_mean = detached_reps.mean(dim=0)
                        batch_std = detached_reps.std(dim=0)

                        # 生成标准正态分布噪声，形状与当前 batch 一致 [B, D]
                        noise = torch.randn_like(detached_reps)

                        # 核心逻辑：利用均值及方差生成特征 (Reparameterization: mu + sigma * noise)
                        generated_ood_reps = batch_mean + batch_std * noise
                    elif 'edl' in cfg['loss_mode']:
                        # 2. 获取随机选取的那个 Affinity Matrix
                        # affinity_mat = input['batch_affinity']
                        affinity_mat = torch.ones(
                            cfg['seen_class_num'],
                            cfg['seen_class_num'],
                            device=cfg['device']
                        )

                        affinity_mat.fill_diagonal_(0)

                        # 3. 调用新的生成函数 (直接利用 rep, 不用 all_means)
                        generated_ood_reps = generate_batch_based_ood(
                            curr_reps.detach(),
                            curr_targets.detach(),
                            affinity_mat.detach(),
                            cfg['device'],
                            fill_diagonal=True if 'mix2' in cfg['loss_mode'] else False
                        )
                        ood_alpha = self.softplus(self.edl_classifier(generated_ood_reps)) + 1.0
                        # print(ood_alpha)

                        # 调用之前加固过的 N.EDL 计算函数
                        # f_c 建议从 cfg 中读取，例如 cfg.get('fisher_c', 0.01)
                        f_c = cfg.get('fisher_c', 0.01)
                        if cfg.get('nedl_server',1):
                            L_nedl_ood = compute_unlabeled_loss_nedl(ood_alpha, fisher_c=f_c,know_ood=False)
                        else:
                            L_nedl_ood = torch.zeros(1, device=cfg['device']).squeeze()

                        # 将该损失加入总损失字典（假设你后续有统一加总逻辑）
                        output['loss'] += L_nedl_ood
                    else:

                        # 2. 获取随机选取的那个 Affinity Matrix
                        affinity_mat = input['batch_affinity']
                        if 'mix2' in cfg['loss_mode']:
                            affinity_mat = torch.ones(
                                cfg['seen_class_num'],
                                cfg['seen_class_num'],
                                device=cfg['device']
                            )

                        # 3. 调用新的生成函数 (直接利用 rep, 不用 all_means)
                        generated_ood_reps = generate_batch_based_ood(
                            curr_reps.detach(),
                            curr_targets.detach(),
                            affinity_mat.detach(),
                            cfg['device'],
                            fill_diagonal=True if 'mix2' in cfg['loss_mode'] else False
                        )

                    # 4. 后续 Loss 计算逻辑 (保持不变)
                    ood_logits = self.classifier(generated_ood_reps)

                    # 定义 OOD 的目标
                    # 如果你有 K+1 类
                    seen_num = cfg['seen_class_num']
                    # ood_targets = torch.full((len(generated_ood_reps),), seen_num, device=cfg['device'], dtype=torch.long)
                    ood_targets = torch.full((len(generated_ood_reps),), seen_num, device=cfg['device']).long()

                    id_logits_part = ood_logits[:, :seen_num]
                    ood_logit_part = ood_logits[:, seen_num:]

                    # ----------------------------------------------------
                    # 这里复用你之前的 Loss 计算逻辑 (Binary Method 或 CE)
                    # ----------------------------------------------------
                    loss_ood = torch.tensor(0.0, device=cfg['device'])

                    if cfg.get('binary_method', None):

                        # === 修改开始: 特殊处理 'detach' 模式 ===
                        if cfg['binary_method'] == 'detach':
                            # 1. 对 ID 部分截断梯度 (作为背景/锚点)
                            id_fixed = id_logits_part.detach()

                            # 2. 重新拼接: [B, K](无梯度) + [B, 1](有梯度) -> [B, K+1]
                            combined_logits = torch.cat([id_fixed, ood_logit_part], dim=1)

                            # 3. 计算标准 K+1 类 CrossEntropy (目标是第 K 类)
                            # 注意：ood_targets 必须是 long 类型 (已在前一步修正)
                            loss_ood = F.cross_entropy(combined_logits, ood_targets)

                        else:
                            # === 其他聚合模式 (Mean, Max, LSE, Shifted) ===
                            if cfg['binary_method'] == 'mean':
                                id_avg_logit = torch.mean(id_logits_part, dim=1, keepdim=True)
                            elif cfg['binary_method'] == 'max':
                                id_avg_logit = torch.max(id_logits_part, dim=1, keepdim=True)
                            elif cfg['binary_method'] == 'lse':
                                T = cfg.get('energy_temp', 1.0)
                                if cfg['data_name'] in ['CIFAR10','SVHN']: T = 1
                                elif cfg['data_name'] in ['CIFAR100']: T = 10
                                correction = T * torch.log(torch.tensor(float(seen_num), device=cfg['device']))
                                id_avg_logit = T * torch.logsumexp(id_logits_part / T, dim=1, keepdim=True) - correction
                            elif cfg['binary_method'] == 'shifted_mean':
                                 id_avg = torch.mean(id_logits_part, dim=1, keepdim=True)
                                 id_max_val, _ = torch.max(id_logits_part, dim=1, keepdim=True)
                                 sharpness_gap = (id_max_val - id_avg).detach()
                                 id_avg_logit = id_avg + sharpness_gap

                            # === 二分类 Loss 计算 (只在非 detach 模式下执行) ===
                            # 构造 [B, 2] 的 Logits: [ID_Score, OOD_Score]
                            binary_logits = torch.cat([id_avg_logit, ood_logit_part], dim=1)
                            # 目标全是 1 (代表 OOD 类)
                            binary_targets = torch.ones(len(generated_ood_reps), device=cfg['device']).long()

                            loss_ood = F.cross_entropy(binary_logits, binary_targets)

                    else:
                        # 默认 K+1 类
                        if ood_logits.shape[1] > seen_num:
                            loss_ood = F.cross_entropy(ood_logits, ood_targets)
                        else:
                            # Energy Loss
                            energy = torch.logsumexp(id_logits_part, dim=1)
                            loss_ood = F.relu(energy - (-5.0)).pow(2).mean()

                    # 添加 Loss
                    if 'weight' in cfg and cfg['weight']:
                        weight = cfg['weight']
                    else:
                        weight = (cfg['target_size'] - cfg['seen_class_num']) / cfg['target_size']
                    # if cfg['data_name'] == 'CIFAR10':
                    # if 'CIFAR10' in cfg['data_name']:
                    #     weight = 1/cfg['seen_class_num']
                    output['loss'] += loss_ood * weight

        return output


def wresnet28x2(momentum=None, track=False):
    data_shape = cfg['data_shape']
    if cfg['task'] == 'OSR':
        target_size = cfg['seen_class_num'] + 1
    else:
        target_size = cfg['target_size']
    depth = cfg['wresnet28x2']['depth']
    widen_factor = cfg['wresnet28x2']['widen_factor']
    drop_rate = cfg['wresnet28x2']['drop_rate']
    model = WideResNet(data_shape, target_size, depth, widen_factor, drop_rate)
    model.apply(init_param)
    model.apply(lambda m: make_batchnorm(m, momentum=momentum, track_running_stats=track))
    # for name, m in model.named_modules():
    #     if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
    #         print(f"{name} track: {m.track_running_stats}")
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
