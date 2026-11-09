import torch
import torch.nn as nn
import torch.nn.functional as F
import models
import numpy as np
from config import cfg
def init_param(m):
    if isinstance(m, nn.Conv2d) and isinstance(m, models.DecConv2d):
        nn.init.kaiming_normal_(m.sigma_weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(m.phi_weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.zero_()
    return m

def make_batchnorm(m, momentum, track_running_stats):
    # if isinstance(m, nn.BatchNorm2d):
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.momentum = momentum
        m.track_running_stats = track_running_stats
        if track_running_stats:
            m.register_buffer('running_mean', torch.zeros(m.num_features, device=m.weight.device))
            m.register_buffer('running_var', torch.ones(m.num_features, device=m.weight.device))
            m.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long, device=m.weight.device))
        else:
            m.running_mean = None
            m.running_var = None
            m.num_batches_tracked = None
    return m


def loss_fn(output, target, reduction='mean'):
    if target.dtype == torch.int64:
        loss = F.cross_entropy(output, target, reduction=reduction)
    else:
        loss = F.mse_loss(output, target, reduction=reduction)
    return loss


def mse_loss(output, target, weight=None):
    mse = F.mse_loss(output, target, reduction='none')
    mse = weight * mse if weight is not None else mse
    mse = torch.sum(mse)
    mse /= output.size(0)
    return mse


def cross_entropy_loss(output, target, weight=None):
    target = (target.topk(1, 1, True, True)[1]).view(-1)
    ce = F.cross_entropy(output, target, reduction='mean', weight=weight)
    return ce


def kld_loss(output, target, weight=None, T=1):
    kld = F.kl_div(F.log_softmax(output, dim=-1), F.softmax(target / T, dim=-1), reduction='none')
    kld = weight * kld if weight is not None else kld
    kld = torch.sum(kld)
    kld /= output.size(0)
    return kld


def mb_sup_loss_openmatch(logits_ova, label):
    batch_size = logits_ova.size(0)
    logits_ova = logits_ova.view(batch_size, 2, -1)
    num_classes = logits_ova.size(2)
    probs_ova = F.softmax(logits_ova, 1)
    label_s_sp = torch.zeros((batch_size, num_classes)).long().to(label.device)
    label_range = torch.arange(0, batch_size).long().to(label.device)
    label_s_sp[label_range[label < num_classes], label[label < num_classes]] = 1
    label_sp_neg = 1 - label_s_sp
    open_loss = torch.mean(torch.sum(-torch.log(probs_ova[:, 1, :] + 1e-8) * label_s_sp, 1))
    open_loss_neg = torch.mean(torch.max(-torch.log(probs_ova[:, 0, :] + 1e-8) * label_sp_neg, 1)[0])
    l_ova_sup = open_loss_neg + open_loss
    return l_ova_sup



# === DivCon 核心组件 ===

def compute_consensus(probs_list):
    """
    Eq. 7: 计算共识度 (Consensus Score)
    输入: list of [B, C] tensors (所有头的预测概率)
    输出: [B] tensor (每个样本的共识度分数，范围 0~1)
    """
    K = len(probs_list)
    B = probs_list[0].size(0)
    sum_exp_dist = torch.zeros(B).to(probs_list[0].device)
    pair_count = 0

    # 计算所有头两两之间的 L1 距离的负指数
    for i in range(K):
        for j in range(i + 1, K):
            # || p_i - p_j ||_1
            l1_dist = torch.norm(probs_list[i] - probs_list[j], p=1, dim=1)
            # exp( - ||...|| )
            sum_exp_dist += torch.exp(-l1_dist)
            pair_count += 1

    if pair_count > 0:
        consensus = sum_exp_dist / pair_count
    else:
        consensus = torch.ones(B).to(probs_list[0].device)

    return consensus

def loss_mi(probs_list, embedding=None):
    """
    Eq. 6 (近似): 互信息损失 L_mi
    目标: 最小化头之间的互信息 -> 鼓励预测多样性 (Diversify)
    实现: 最大化头预测分布之间的 Jensen-Shannon 散度或成对 KL 散度
    """
    loss_div = torch.tensor(0.0).to(probs_list[0].device)
    K = len(probs_list)
    if K < 2:
        return loss_div

    # 论文中使用互信息，实际工程中常用 "Pairwise KL Divergence" 的负值来实现
    # 即：让 Head A 的分布尽可能远离 Head B 的分布
    for i in range(K):
        for j in range(i + 1, K):
            p_i = probs_list[i]
            p_j = probs_list[j]

            # 为了数值稳定性，加上微小量
            p_i = torch.clamp(p_i, min=1e-7, max=1.0)
            p_j = torch.clamp(p_j, min=1e-7, max=1.0)

            # 计算对称 KL 散度 (SymKL)
            # KL(P||Q) = sum(P * log(P/Q))
            kl_i_j = F.kl_div(torch.log(p_j), p_i, reduction='batchmean')
            kl_j_i = F.kl_div(torch.log(p_i), p_j, reduction='batchmean')

            # 我们希望差异最大化，即 loss 最小化，所以取负号
            # 这种 loss 会推着两个分布分开
            loss_div -= (kl_i_j + kl_j_i) / 2

    # 归一化 loss
    return loss_div / (K * (K - 1) / 2)




def compute_ood_proto_loss(
    id_reps,                 # [N_id, D]
    ood_reps,                # [N_proto, D]
    proto_logits,            # [N_proto, C + 1]
    num_known_classes,       # C
    device,
):
    """
    使用：
    - ID 子空间重构误差 Top-K（几何最远）
    - 已知类熵 Top-K（分布最平）
    二者交集作为 OOD set
    """

    # 固定比例 Top-K（一半）
    num_proto = ood_reps.size(0)
    k_svd = (num_proto*3) // 4
    k_entropy = (num_proto*3) // 4

    id_reps = id_reps.to(device)
    ood_reps = ood_reps.to(device)
    proto_logits = proto_logits.to(device)

    # ============================================================
    # Step 1. ID 子空间（SVD，无能量截断）
    # ============================================================
    proto_recon_errors = torch.zeros(num_proto, device=device)

    if id_reps.size(0) > 1:
        id_mean = id_reps.mean(dim=0)
        id_centered = id_reps - id_mean

        try:
            _, _, Vh = torch.linalg.svd(id_centered, full_matrices=False)
        except RuntimeError:
            _, _, V = torch.svd(id_centered)
            Vh = V.t()

        # 直接使用完整 ID 子空间
        basis = Vh  # [r, D]

        def recon_error(x):
            centered = x - id_mean
            weight = centered @ basis.t()
            recon = weight @ basis
            return torch.norm(centered - recon, dim=1)

        proto_recon_errors = recon_error(ood_reps)

    # ============================================================
    # Step 2. 已知类熵（Entropy over visible classes）
    # ============================================================
    known_logits = proto_logits[:, :num_known_classes]
    known_probs = F.softmax(known_logits, dim=1)

    entropy_scores = -torch.sum(
        known_probs * torch.log(known_probs + 1e-12),
        dim=1
    )   # 越大越像 OOD

    # ============================================================
    # Step 3. Top-K 选择 & 交集
    # ============================================================
    _, idx_svd = torch.topk(proto_recon_errors, k=k_svd)
    _, idx_entropy = torch.topk(entropy_scores, k=k_entropy)

    mask_svd = torch.zeros(num_proto, device=device, dtype=torch.bool)
    mask_entropy = torch.zeros_like(mask_svd)

    mask_svd[idx_svd] = True
    mask_entropy[idx_entropy] = True

    final_ood_mask = mask_svd # & mask_entropy
    print('server_mask_ratoio:',final_ood_mask.sum(),len(final_ood_mask))

    # ============================================================
    # Step 4. OOD loss
    # ============================================================
    ood_label = torch.full(
        (num_proto,),
        num_known_classes,
        device=device,
        dtype=torch.long
    )

    loss_per_sample = loss_fn(proto_logits, ood_label, reduction='none')

    if final_ood_mask.sum() > 0:
        loss = (loss_per_sample * final_ood_mask).sum() / final_ood_mask.sum()
    else:
        loss = torch.tensor(0.0, device=device)

    return loss, final_ood_mask

def generate_batch_based_ood(reps, targets, affinity_matrix, device, mix_ratio=1.0, fill_diagonal=False):
    """
    基于当前 Batch 的特征和选定的 Affinity Matrix 生成 OOD。
    仅在当前 Batch 存在的类别之间进行 Mixup。
    """
    # 1. 获取当前 Batch 中存在的唯一类别
    unique_classes = torch.unique(targets)

    # 如果当前 Batch 类别太少，直接返回噪声或原特征
    if len(unique_classes) < 2:
        return torch.randn_like(reps)[:len(reps)]

    # 2. 构建类别到样本索引的映射
    class_to_indices = {}
    for c in unique_classes:
        class_to_indices[c.item()] = (targets == c).nonzero(as_tuple=True)[0]

    # 3. 对 Affinity Matrix 进行掩码
    mask = torch.zeros_like(affinity_matrix, dtype=torch.bool)
    try:
        grid_x, grid_y = torch.meshgrid(unique_classes, unique_classes, indexing='ij')
    except TypeError:
        grid_x, grid_y = torch.meshgrid(unique_classes, unique_classes)
    mask[grid_x, grid_y] = True
    if not fill_diagonal:
        mask.fill_diagonal_(False)

    # 4. 获取采样概率
    masked_affinity = affinity_matrix * mask.float()
    total_affinity = masked_affinity.sum()
    if total_affinity > 0:
        probs = masked_affinity.view(-1) / total_affinity
    else:
        probs = mask.view(-1).float() / mask.sum()

    # 5. 生成样本数量
    num_gen = reps.size(0)

    # 6. 采样类别对
    pair_indices = torch.multinomial(probs, num_gen, replacement=True)
    num_classes = affinity_matrix.shape[0]

    class_a_list = pair_indices // num_classes
    class_b_list = pair_indices % num_classes

    # 7. 收集特征
    # 生成混合系数 [N, 1]
    # if 'edl' in cfg['loss_mode']:
    #     # 策略 1: 直接固定 0.5，绝对的真空带中心
    #     lam = torch.full((num_gen, 1), 0.5).to(device)

    #     # 策略 2: 在 0.5 附近小范围抖动 (0.45 ~ 0.55)，增加一点泛化性
    #     # lam = 0.45 + 0.1 * torch.rand((num_gen, 1)).to(device)
    # else:
    #     # 原有的 Beta 分布 (Beta(2,2) 已经比 Beta(0.2,0.2) 保守，但仍可能接近边界)
    #     lam = torch.distributions.Beta(2.0, 2.0).sample((num_gen, 1)).to(device)
    lam = torch.distributions.Beta(2.0, 2.0).sample((num_gen, 1)).to(device)
    feats_a = []
    feats_b = []

    for i in range(num_gen):
        c_a = class_a_list[i].item()
        c_b = class_b_list[i].item()

        # [修改点 1] 使用 .item() 获取标量索引
        # 这样 reps[scalar_idx] 返回的是 [D] 而不是 [1, D]
        idx_tensor_a = class_to_indices[c_a][torch.randint(len(class_to_indices[c_a]), (1,))]
        idx_a = idx_tensor_a.item()

        idx_tensor_b = class_to_indices[c_b][torch.randint(len(class_to_indices[c_b]), (1,))]
        idx_b = idx_tensor_b.item()

        feats_a.append(reps[idx_a])
        feats_b.append(reps[idx_b])

    # [修改点 2] 此时 stack 结果为 [N, D]
    feats_a = torch.stack(feats_a) # [10, 128]
    feats_b = torch.stack(feats_b) # [10, 128]

    # 8. 执行 Mixup
    # [10, 1] * [10, 128] + [10, 1] * [10, 128] -> [10, 128]
    # 维度完美匹配，不再广播出由 [10, 10, 128]
    mixed_features = lam * feats_a + (1 - lam) * feats_b
    # ================= [新增逻辑：时间复杂度较低的非硬阈值过滤] =================
    if False:
        with torch.no_grad():
            # 1. 归一化特征
            reps_norm = F.normalize(reps, p=2, dim=1)
            mixed_norm = F.normalize(mixed_features, p=2, dim=1)

            # 2. 计算每个类内部的平均相似度 (Intra-class Similarity)
            # 我们需要知道每个类自己有多“紧凑”
            class_internal_sims = {}
            unique_batches = torch.unique(targets)
            for c in unique_batches:
                c_mask = (targets == c)
                if c_mask.sum() > 1:
                    c_reps = reps_norm[c_mask]
                    # 计算该类内部两两相似度
                    sim_matrix = torch.mm(c_reps, c_reps.t())
                    # 排除对角线自身相似度 1.0
                    n = c_reps.size(0)
                    internal_sim = (sim_matrix.sum() - n) / (n * (n - 1))
                    class_internal_sims[c.item()] = internal_sim
                else:
                    # 样本太少无法计算内部相似度时，给个全局默认值或 0.8
                    class_internal_sims[c.item()] = torch.tensor(0.8).to(device)

            # 3. 找到每个生成的 OOD 离它最近的那个 ID 类是谁，以及相似度是多少
            # sim_ood_id 维度: [num_gen, batch_size]
            sim_ood_id = torch.mm(mixed_norm, reps_norm.t())
            # max_sim: 最高的相似度值, nearest_idx: 对应的 ID 样本索引
            max_sim, nearest_idx = torch.max(sim_ood_id, dim=1)

            # 4. 执行逐样本审查
            # 获取每个 OOD 样本最近 ID 样本所属的类别
            nearest_class = targets[nearest_idx]

            # 构造一个对应每个 OOD 样本的阈值向量
            # 逻辑：生成的 OOD 与最近 ID 的相似度，必须小于该 ID 类自身的内部相似度
            thresholds = torch.tensor([class_internal_sims[c.item()] for c in nearest_class]).to(device)

            keep_mask = max_sim < thresholds

        # 如果过滤后剩下的样本太少（比如全被过滤了），回退到全部保留但加一点噪声
        # print(keep_mask.sum())
        # print(num_gen,keep_mask.sum())
        if keep_mask.sum() < (num_gen // 4):
            return mixed_features + torch.randn_like(mixed_features) * 0.05


        return mixed_features[keep_mask]
    # =========================================================================

    return mixed_features

class BDMatchAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.linear_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, query_embed):
        B, N, C = x.shape
        K = query_embed.size(1)
        # Attention logic to generate label-specific features
        q = self.linear_q(query_embed).expand(B, -1, -1).reshape(B, K, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.linear_k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, K, C)
        x = self.proj_drop(x)
        return x
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

def compute_labeled_loss_pedl(evi_alp, labels, epoch, target_concentration=100, fisher_c=0.01,know_id=True):
    # 仅取有标签部分的数据
    b_size = labels.size(0)
    # alp = evi_alp[:b_size]
    alp = torch.clamp(evi_alp[:b_size], min=1.01)
    alp0 = torch.sum(alp, dim=-1, keepdim=True)

    num_classes = alp.size(1)
    labels_1hot = torch.zeros_like(alp).scatter_(1, labels.view(-1, 1), 1.0)
    # 论文中通常会给 labels 做微小的平滑处理，代码里体现为 + 1/num
    # labels_1hot += 1.0 / num_classes

    gamma1_alp = torch.polygamma(1, alp)
    gamma1_alp0 = torch.polygamma(1, alp0)

    # 1. MSE Loss (P.EDL的核心): 预测均值与真值的差距
    gap = labels_1hot - alp / alp0
    # loss_mse = (gap.pow(2) * gamma1_alp).sum(-1).mean()
    if know_id:
        loss_mse = (gap.pow(2)).sum(-1).mean()
        term2 = 1.0 - (gamma1_alp0 / (gamma1_alp + 1e-8)).sum(-1)
        loss_det_fisher =  (torch.log(gamma1_alp + 1e-8).sum(-1) +
                    torch.log(torch.clamp(term2, min=1e-7))).mean()
        loss_var = (alp * (alp0 - alp) /
                (alp0 * alp0 * (alp0 + 1))).sum(-1).mean() / 3.0
    else:
        if cfg['use_3sigma'] == False:
            loss_mse = (gap.pow(2)).sum(-1).mean()
            loss_var = (alp * (alp0 - alp) /
                (alp0 * alp0 * (alp0 + 1))).sum(-1).mean() / 3.0
        else:
            mode = cfg.get('weight_mode')
            if mode == 'simple':
                weight = 1.0 / (alp + 1e-6)
            elif mode == 'exp':
                weight = torch.exp(-alp + 1.0)
            elif mode == 'soft':
                weight = 1.0 / (1.0 + 0.1 * alp)
            elif mode == 'gauss':
                weight = torch.exp(-alp.pow(2) / 2.0)
            elif mode == 'log':
                weight = torch.log(1.0 + 1.0 / (alp + 1e-6))
            else:
                weight = gamma1_alp

            loss_mse = (gap.pow(2) * weight).sum(-1).mean()
            loss_var = (alp * (alp0 - alp) * weight / (alp0 * alp0 * (alp0 + 1))).sum(-1).mean() / 3.0
        # else:
        #     loss_mse = (gap.pow(2) * gamma1_alp).sum(-1).mean()
        #     loss_var = (alp * (alp0 - alp) * gamma1_alp /
        #         (alp0 * alp0 * (alp0 + 1))).sum(-1).mean() / 3.0
        if cfg['nouse_fisher']:
            loss_det_fisher = torch.zeros(1, device=alp.device).squeeze()
        else:
            term2 = 1.0 - (gamma1_alp0 / (gamma1_alp + 1e-8)).sum(-1)
            loss_det_fisher =  (torch.log(gamma1_alp + 1e-8).sum(-1) +
                        torch.log(torch.clamp(term2, min=1e-7))).mean()
        # loss_var = (alp * (alp0 - alp) * gamma1_alp /
        #         (alp0 * alp0 * (alp0 + 1))).sum(-1).mean() / 3.0

    # 2. Variance Loss: 进一步减小预测的不确定性方差
    # loss_var = (alp * (alp0 - alp) * gamma1_alp /
    #             (alp0 * alp0 * (alp0 + 1))).sum(-1).mean() / 3.0

    # loss_det_fisher = - (torch.log(gamma1_alp).sum(-1) +
    #             torch.log(1.0 - (gamma1_alp0 / gamma1_alp).sum(-1))).mean()
    # 为 Fisher 项增加微小的 epsilon，并防止 log 负数


    # 3. Strengthened KL Loss: 引导证据量向 target_concentration (100) 靠拢
    # 注意：这里需要一个全1的mask，因为全是有效样本
    mask = torch.ones(b_size).to(cfg['device'])
    # loss_kl = compute_kl_loss(alp, labels, target_concentration, mask)
    # 1. 构造目标：非真值类设为 1.0，真值类填回 alp 自身（扣除该类影响）
    target_alpha_kl = torch.ones_like(alp).to(cfg['device'])
    current_alp_true = torch.gather(alp, 1, labels.view(-1, 1))
    target_alpha_kl.scatter_(1, labels.view(-1, 1), current_alp_true)

    # 2. 计算通用 KL 散度（真值类项将自动抵消为 0）
    alp0 = torch.sum(alp, dim=-1, keepdim=True)
    target_alp0 = torch.sum(target_alpha_kl, dim=-1, keepdim=True)

    alp0_term = torch.lgamma(alp0 + 1e-8) - torch.lgamma(target_alp0 + 1e-8)
    alphas_term = torch.sum(
        torch.lgamma(target_alpha_kl + 1e-8) - torch.lgamma(alp + 1e-8)
        + (alp - target_alpha_kl) * (torch.digamma(alp + 1e-8) - torch.digamma(alp0 + 1e-8)),
        dim=-1, keepdim=True
    )

    loss_kl = (torch.squeeze(alp0_term + alphas_term)).mean()

    # 退火系数
    regr = np.minimum(1.0, (epoch + 1) / 10.)

    # 组合 L_P_EDL + L_KL (对应论文公式8的前半部分)
    # loss_p_edl = loss_mse + loss_var + regr * 0.05 * loss_kl + fisher_c * loss_det_fisher
    loss_p_edl = loss_mse + loss_var + fisher_c * loss_det_fisher
    # print('labeled:',loss_mse.item(),loss_det_fisher.item(),loss_kl.item(),loss_var.item())
    # print(f"\n>>> [Epoch {epoch}] Labeled Sample Distribution Analysis (First 5 samples) <<<")

    # # 将 Tensor 转回 CPU 以便处理
    # with torch.no_grad():
    #     # alp 是 Dirichlet 的参数 alpha
    #     probs = alp / alp0  # 预测的概率均值
    #     uncertainty = num_classes / alp0.squeeze() # 整体不确定性 u = K/S

    #     for i in range(min(5, b_size)):
    #         target = labels[i].item()
    #         pred_class = torch.argmax(probs[i]).item()

    #         # 格式化打印：显示该样本的 alpha 向量（证据分布）
    #         # 我们保留 2 位小数
    #         alp_vec = alp[i].detach().cpu().numpy()
    #         formatted_alp = [f"{x:.2f}" for x in alp_vec]

    #         print(f"Sample {i}:")
    #         print(f"  - Real Label: {target} | Predicted: {pred_class}")
    #         print(f"  - Alpha Vector: {formatted_alp}")
    #         print(f"  - Max Alpha: {alp_vec.max():.2f} | Uncertainty(u): {uncertainty[i].item():.4f}")
    #         print("-" * 50)
    return loss_p_edl


def compute_unlabeled_loss_nedl(evi_alp_unlabeled, fisher_c=0.01,know_ood=False):
    """
    针对无标签数据的负向证据损失 (N.EDL) + KL 散度约束
    目标：抑制证据产生，使 alpha 趋向于 1 (即完全不确定状态)
    """
    alp = evi_alp_unlabeled
    alp0 = torch.sum(alp, dim=-1, keepdim=True)
    num_classes = alp.size(1)

    # 1. 负向 MSE 目标：均匀分布
    labels_uniform = torch.ones_like(alp) / num_classes
    gamma1_alp = torch.polygamma(1, alp)
    gamma1_alp0 = torch.polygamma(1, alp0)

    gap = labels_uniform - alp / alp0
    # if know_ood and not cfg['adj_fish']:
    #     loss_mse_neg = (gap.pow(2)).sum(-1).mean()
    # else:
    if cfg['use_3sigma'] == False:
        loss_mse_neg = (gap.pow(2)).sum(-1).mean()
    else:
        # 增加一个控制逻辑：如果开启了复杂容错，则根据 mode 切换
        mode = cfg.get('weight_mode')
        if mode == 'simple':
            weight = 1.0 / (alp + 1e-6)
        elif mode == 'exp':
            weight = torch.exp(-alp + 1.0)
        elif mode == 'soft':
            weight = 1.0 / (1.0 + 0.1 * alp)
        elif mode == 'gauss':
            weight = torch.exp(-alp.pow(2) / 2.0)
        elif mode == 'log':
            weight = torch.log(1.0 + 1.0 / (alp + 1e-6))
        else:
            weight = gamma1_alp
        
        loss_mse_neg = (gap.pow(2) * weight).sum(-1).mean()

    # else:
    #     loss_mse_neg = (gap.pow(2) * gamma1_alp).sum(-1).mean()

    # 2. Fisher 信息项：几何约束压制
    # loss_det_fisher = - (torch.log(gamma1_alp).sum(-1) +
    #                      torch.log(1.0 - (gamma1_alp0 / gamma1_alp).sum(-1))).mean()
    eps = 1e-8
    term_fisher_inner = 1.0 - (gamma1_alp0 / (gamma1_alp + eps)).sum(-1)

    # 3. 无标签 KL 损失 (关键修改)
    # 目标：将无标签样本推向“无证据”状态（alpha=1）
    # 这里的 target_concentration 必须设为 1.0，代表均匀的狄利克雷先验
    mask_u = torch.ones(alp.size(0)).to(cfg['device'])

    # 我们为无标签数据创建一个虚拟的标签（实际上不起作用，因为 concentration=1.0 时
    # target_alphas 全为 1，scatter 操作不改变结果）
    dummy_labels = torch.zeros(alp.size(0), dtype=torch.long).to(cfg['device'])
    if know_ood:
        if cfg['nouse_kl']:
            loss_kl_unlabeled = torch.zeros(1,device=alp.device).squeeze()
        else:
            loss_kl_unlabeled = compute_kl_loss(alp, dummy_labels, target_concentration=1.0, mask=mask_u)
        if cfg['adj_fish']:
            loss_det_fisher = - (
                torch.log(gamma1_alp + eps).sum(-1) +
                torch.log(torch.clamp(term_fisher_inner, min=1e-7))
            ).mean()
        else:
            loss_det_fisher = torch.zeros(1, device=alp.device).squeeze()
    else:
        if cfg['use_kl']==1:
            loss_kl_unlabeled = compute_kl_loss(alp, dummy_labels, target_concentration=1.0, mask=mask_u)
            loss_det_fisher = - (
                torch.log(gamma1_alp + eps).sum(-1) +
                torch.log(torch.clamp(term_fisher_inner, min=1e-7))
            ).mean()
        elif cfg['use_kl']==2:
            loss_kl_unlabeled = torch.zeros(1,device=alp.device).squeeze()
            loss_det_fisher = - (
                torch.log(gamma1_alp + eps).sum(-1) +
                torch.log(torch.clamp(term_fisher_inner, min=1e-7))
            ).mean()
        elif cfg['nouse_kl']:
            loss_kl_unlabeled = torch.zeros(1,device=alp.device).squeeze()
            loss_det_fisher = - (
                torch.log(gamma1_alp + eps).sum(-1) +
                torch.log(torch.clamp(term_fisher_inner, min=1e-7))
            ).mean()
        else:
            loss_kl_unlabeled = torch.zeros(1,device=alp.device).squeeze()
            loss_det_fisher = torch.zeros(1, device=alp.device).squeeze()


    # 组合损失：MSE抑制 + Fisher压制 + KL收缩
    # 注意：无标签 KL 的权重通常较小，起到正则化作用
    # loss_n_edl = loss_mse_neg + fisher_c * loss_det_fisher + 0.05 * loss_kl_unlabeled
    kl_w = 0.05 if cfg.get('kl_w',0.05) == 0 else 0.05
    loss_n_edl = loss_mse_neg + kl_w * loss_kl_unlabeled + fisher_c * loss_det_fisher
    # print('unlabeled:',loss_mse_neg.item(),loss_det_fisher.item(),loss_kl_unlabeled.item())
    return loss_n_edl

def compute_kl_loss(alphas, labels, target_concentration, mask, concentration=1.0, epsilon=1e-8):
    # TODO: Need to make sure this actually works right...
    # todo: so that concentration is either fixed, or on a per-example setup
    # Create array of target (desired) concentration parameters

    if target_concentration < 1.0:
        concentration = target_concentration

    target_alphas = torch.ones_like(alphas) * concentration
    target_alphas += torch.zeros_like(alphas).scatter_(-1, labels.unsqueeze(-1), target_concentration - 1)

    alp0 = torch.sum(alphas, dim=-1, keepdim=True)
    target_alp0 = torch.sum(target_alphas, dim=-1, keepdim=True)

    alp0_term = torch.lgamma(alp0 + epsilon) - torch.lgamma(target_alp0 + epsilon)
    alp0_term = torch.where(torch.isfinite(alp0_term), alp0_term, torch.zeros_like(alp0_term))
    assert torch.all(torch.isfinite(alp0_term)).item()

    alphas_term = torch.sum(torch.lgamma(target_alphas + epsilon) - torch.lgamma(alphas + epsilon)
                            + (alphas - target_alphas) * (torch.digamma(alphas + epsilon) -
                                                            torch.digamma(alp0 + epsilon)), dim=-1, keepdim=True)
    alphas_term = torch.where(torch.isfinite(alphas_term), alphas_term, torch.zeros_like(alphas_term))
    assert torch.all(torch.isfinite(alphas_term)).item()

    loss = (torch.squeeze(alp0_term + alphas_term) * mask).mean()

    return loss