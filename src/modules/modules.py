import copy
import datetime
import numpy as np
import sys
import time
import torch
import torch.nn.functional as F
import models
from itertools import compress
from config import cfg
from data import make_data_loader, make_batchnorm_stats, FixTransform, MixDataset
from utils import to_device, make_optimizer, collate, to_device, OptionalNoGrad
from metrics import Accuracy
from collections import deque, defaultdict
import random

class Server:
    def __init__(self, model):
        self.model_state_dict = save_model_state_dict(model.state_dict())
        if 'fmatch' in cfg['loss_mode']:
            optimizer = make_optimizer(model.make_sigma_parameters(), 'local')
            global_optimizer = make_optimizer(model.make_phi_parameters(), 'global')
        else:
            optimizer = make_optimizer(model.parameters(), 'local')
            global_optimizer = make_optimizer(model.parameters(), 'global')
        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
        if cfg['task'] == 'OSR':
            self.prototypes = []
            self.ood_threshold_mean = 0
            self.beta = torch.distributions.beta.Beta(torch.tensor([cfg['alpha']]), torch.tensor([cfg['alpha']]))
            if 'promax' in cfg['sel_method']:
                self.kernels = None
            if 'subspaces' in cfg['sel_method']:
                self.id_bases = None
                self.id_means = None
                self.ood_bases = []
                self.ood_means = []
                D = 128 if cfg['data_name'] not in ['TinyImageNet','MIOTCD'] else 256 # 特征维度
                self.all_means = torch.zeros(cfg['seen_class_num'], D)
                self.all_vars = torch.ones(cfg['seen_class_num'], D) # 默认方差为1
                self.persistent_class_mask = torch.zeros(cfg['seen_class_num'], dtype=torch.bool) # 记录哪些类已经由于数据了
            if 'affinity' in cfg['sel_method']:
                self.affinitys = None

    def distribute(self, client, batchnorm_dataset=None):
        model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        bn_stats_present = any('running_mean' in k or 'running_var' in k
                       for k in self.model_state_dict.keys())
        if bn_stats_present:
            model.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=True))
        model.load_state_dict(self.model_state_dict)
        if bn_stats_present:
            model.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=False))
        if batchnorm_dataset is not None:
            model = make_batchnorm_stats(batchnorm_dataset, model, 'global')
        model_state_dict = save_model_state_dict(model.state_dict())
        for m in range(len(client)):
            if client[m].active:
                client[m].model_state_dict = copy.deepcopy(model_state_dict)
                if 'promax' in cfg['sel_method'] and cfg['task'] == 'OSR':
                    client[m].kernels = self.kernels
                if 'subspaces' in cfg['sel_method'] and getattr(self, 'id_bases', None) is not None and cfg['task'] == 'OSR':
                    client[m].id_bases = self.id_bases
                    client[m].id_means = self.id_means
                    client[m].ood_bases = self.ood_bases
                    client[m].ood_means = self.ood_means
                    client[m].id_vars = self.id_vars
                if client[m].optimizer_state_dict is None:
                    client[m].optimizer_state_dict = self.optimizer_state_dict
            else:
                # === 未选中：主动释放内存/置空 ===
                # 这非常安全，能大幅减小保存文件的大小
                client[m].model_state_dict = None
                client[m].optimizer_state_dict = None
        return

    def update(self, client):
        if 'fmatch' not in cfg['loss_mode']:
            with torch.no_grad():
                valid_client = [client[i] for i in range(len(client)) if client[i].active]
                if len(valid_client) > 0:
                    model = eval('models.{}()'.format(cfg['model_name']))
                    bn_stats_present = any('running_mean' in k or 'running_var' in k
                       for k in self.model_state_dict.keys())
                    if bn_stats_present:
                        model.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=True))
                    model.load_state_dict(self.model_state_dict)
                    if bn_stats_present:
                        model.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=False))
                    global_optimizer = make_optimizer(model.parameters(), 'global')
                    global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                    global_optimizer.zero_grad()
                    weight = torch.ones(len(valid_client))
                    weight = weight / weight.sum()
                    for k, v in model.named_parameters():
                        parameter_type = k.split('.')[-1]
                        if 'weight' in parameter_type or 'bias' in parameter_type:
                            tmp_v = v.data.new_zeros(v.size())
                            for m in range(len(valid_client)):
                                tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                            v.grad = (v.data - tmp_v).detach()
                    global_optimizer.step()
                    self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                    self.model_state_dict = save_model_state_dict(model.state_dict())
                    if cfg['task'] == 'OSR':
                        self.prototypes = []
                        self.ood_threshold_mean = 0
                        if 'subspaces' in cfg['sel_method']:
                            self.ood_bases = []
                        if 'affinity' in cfg['sel_method']:
                            self.affinitys = []
                        for m in range(len(valid_client)):
                            if valid_client[m].prototype is not None:
                                self.prototypes.append(copy.deepcopy(valid_client[m].prototype))
                            self.ood_threshold_mean += valid_client[m].ood_threshold_mean
                            if 'subspaces' in cfg['sel_method'] and len(valid_client[m].ood_means) > 0:
                                self.ood_bases.append(copy.deepcopy(valid_client[m].ood_bases))
                                self.ood_means.append(copy.deepcopy(valid_client[m].ood_means))
                            if 'affinity' in cfg['sel_method']:
                                self.affinitys.append(copy.deepcopy(valid_client[m].affinity_matrix))
                                valid_client[m].affinity_matrix = None
                        self.ood_threshold_mean/=len(valid_client)
                        print('ood_threshold_SERVER:',self.ood_threshold_mean)
                        if 'subspaces' in cfg['sel_method'] and len(self.ood_bases) > 0:
                            # 清空之前的常规原型，准备装入所有基向量
                            self.prototypes = []

                            # 遍历收集到的所有客户端的基底矩阵
                            # batch_bases 的形状是 [Rank, D] (例如 [8, 512])
                            # print(self.ood_means)
                            for batch_bases, batch_mean in zip(self.ood_bases, self.ood_means):

                                # 1. 确保维度匹配以便广播 (Broadcasting)
                                # 如果 batch_mean 是 [D]，需要扩展为 [1, D] 以便和 [R, D] 相加
                                # print(batch_mean)
                                if batch_mean.dim() == 1:
                                    batch_mean = batch_mean.unsqueeze(0)

                                # 2. 核心修正：合成虚拟样本 (Synthesize Virtual Samples)
                                # 几何意义：从中心沿着主要变化方向移动 1 个单位
                                # Shape: [R, D] = [1, D] + [R, D]
                                virtual_samples = batch_mean.cpu() + batch_bases.cpu()

                                # 3. (可选) 是否添加反方向的点?
                                # virtual_samples_neg = batch_mean - batch_bases

                                # 4. (可选) 是否添加均值本身?
                                # self.prototypes.append(batch_mean.squeeze(0))

                                # 5. 将合成的点加入原型列表
                                self.prototypes.extend(torch.unbind(virtual_samples, dim=0))

                            print(f"Server: Subspaces active. Collected {len(self.prototypes)} OOD basis vectors in list.")
        elif 'fmatch' in cfg['loss_mode']:
            with torch.no_grad():
                valid_client = [client[i] for i in range(len(client)) if client[i].active]
                if len(valid_client) > 0:
                    model = eval('models.{}()'.format(cfg['model_name']))
                    model.load_state_dict(self.model_state_dict)
                    global_optimizer = make_optimizer(model.make_phi_parameters(), 'global')
                    global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                    global_optimizer.zero_grad()
                    weight = torch.ones(len(valid_client))
                    weight = weight / weight.sum()
                    for k, v in model.named_parameters():
                        parameter_type = k.split('.')[-1]
                        if 'weight' in parameter_type or 'bias' in parameter_type:
                            tmp_v = v.data.new_zeros(v.size())
                            for m in range(len(valid_client)):
                                tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                            v.grad = (v.data - tmp_v).detach()
                    global_optimizer.step()
                    self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                    self.model_state_dict = save_model_state_dict(model.state_dict())
                    if cfg['task'] == 'OSR':
                        self.prototypes = []
                        self.ood_threshold_mean = 0
                        for m in range(len(valid_client)):
                            if valid_client[m].prototype is not None:
                                self.prototypes.append(copy.deepcopy(valid_client[m].prototype))
                                self.ood_threshold_mean += valid_client[m].ood_threshold_mean
                                if 'subspaces' in cfg['sel_method']:
                                    self.ood_bases.append(copy.deepcopy(valid_client[m].ood_bases))
                                    self.ood_means.append(copy.deepcopy(valid_client[m].ood_means))
                        self.ood_threshold_mean/=len(valid_client)
                        print('ood_threshold_SERVER:',self.ood_threshold_mean)
        else:
            raise ValueError('Not valid loss mode')

        num_classes = cfg['seen_class_num']
        D = 0
        for c_obj in valid_client:
            if hasattr(c_obj, 'all_means') and c_obj.all_means is not None:
                D = c_obj.all_means.size(1)
                break

        if D > 0:
            # 初始化累加器
            global_sum_means = torch.zeros(num_classes, D)
            global_sum_vars = torch.zeros(num_classes, D)
            # 记录有多少个客户端拥有该类别的数据
            global_client_counts = torch.zeros(num_classes)

            # 2. 遍历客户端进行累加
            for m in range(len(valid_client)):
                vc = valid_client[m]
                if hasattr(vc, 'all_means') and vc.all_means is not None:
                    c_means = vc.all_means.cpu()
                    c_vars = vc.all_vars.cpu()

                    # 判断客户端是否有该类别的有效数据
                    # 利用 class_counts > 0 作为布尔掩码
                    # 如果之前Client代码没存class_counts，可以用 c_vars.sum(dim=1) > 0 判断
                    if hasattr(vc, 'class_counts'):
                        has_data_mask = (vc.class_counts.cpu() > 0).float()
                    else:
                        # 兜底逻辑：如果方差和不为0，认为有数据
                        has_data_mask = (c_vars.sum(dim=1) > 1e-6).float()

                    # 扩展掩码以便广播 [C] -> [C, 1]
                    mask_expanded = has_data_mask.unsqueeze(1)

                    # 累加 (只有有数据的类别才会被加上)
                    global_sum_means += c_means * mask_expanded
                    global_sum_vars += c_vars * mask_expanded
                    global_client_counts += has_data_mask


            # 避免除以 0
            valid_classes_mask = global_client_counts > 0

            if valid_classes_mask.any():
                div_counts = global_client_counts[valid_classes_mask].unsqueeze(1)

                # 均值 = 总和 / 客户端数量
                self.all_means[valid_classes_mask] = global_sum_means[valid_classes_mask] / div_counts
                self.all_vars[valid_classes_mask] = global_sum_vars[valid_classes_mask] / div_counts
                valid_classes_mask = global_client_counts > 0

                # [新增] 记录当前覆盖了多少个类别
                self.persistent_class_mask = self.persistent_class_mask | valid_classes_mask

        for i in range(len(client)):
            client[i].active = False
        return

    def update_parallel(self, client):
        if 'frgd' not in cfg['loss_mode']:
            with torch.no_grad():
                valid_client_server = [self] + [client[i] for i in range(len(client)) if client[i].active]
                if len(valid_client_server) > 0:
                    model = eval('models.{}()'.format(cfg['model_name']))
                    model.load_state_dict(self.model_state_dict)
                    global_optimizer = make_optimizer(model.parameters(), 'global')
                    global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                    global_optimizer.zero_grad()
                    weight = torch.ones(len(valid_client_server))
                    weight = weight / (2 * (weight.sum() - 1))
                    weight[0] = 1 / 2 if len(valid_client_server) > 1 else 1
                    for k, v in model.named_parameters():
                        parameter_type = k.split('.')[-1]
                        if 'weight' in parameter_type or 'bias' in parameter_type:
                            tmp_v = v.data.new_zeros(v.size())
                            for m in range(len(valid_client_server)):
                                tmp_v += weight[m] * valid_client_server[m].model_state_dict[k]
                            v.grad = (v.data - tmp_v).detach()
                    global_optimizer.step()
                    self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                    self.model_state_dict = save_model_state_dict(model.state_dict())
        elif 'frgd' in cfg['loss_mode']:
            with torch.no_grad():
                valid_client_server = [self] + [client[i] for i in range(len(client)) if client[i].active]
                num_valid_client = len(valid_client_server) - 1
                if len(valid_client_server) > 0:
                    model = eval('models.{}()'.format(cfg['model_name']))
                    model.load_state_dict(self.model_state_dict)
                    global_optimizer = make_optimizer(model.parameters(), 'global')
                    global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                    global_optimizer.zero_grad()
                    weight = torch.ones(len(valid_client_server)) / (num_valid_client // 2 + 1)
                    for k, v in model.named_parameters():
                        parameter_type = k.split('.')[-1]
                        if 'weight' in parameter_type or 'bias' in parameter_type:
                            tmp_v_1 = v.data.new_zeros(v.size())
                            tmp_v_1 += weight[0] * valid_client_server[0].model_state_dict[k]
                            for m in range(1, num_valid_client // 2 + 1):
                                tmp_v_1 += weight[m] * valid_client_server[m].model_state_dict[k]
                            tmp_v_2 = v.data.new_zeros(v.size())
                            tmp_v_2 += weight[0] * valid_client_server[0].model_state_dict[k]
                            for m in range(num_valid_client // 2 + 1, len(valid_client_server)):
                                tmp_v_2 += weight[m] * valid_client_server[m].model_state_dict[k]
                            tmp_v = (tmp_v_1 + tmp_v_2) / 2
                            v.grad = (v.data - tmp_v).detach()
                    global_optimizer.step()
                    self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                    self.model_state_dict = save_model_state_dict(model.state_dict())
        else:
            raise ValueError('Not valid loss mode')
        for i in range(len(client)):
            client[i].active = False
        return


    def check_subspace_orthogonality(self):
        """
        检查服务器生成的 ID 子空间之间的正交性（差异大小）。
        """
        if not hasattr(self, 'id_bases') or self.id_bases is None:
            print("Server has no ID bases yet.")
            return

        with torch.no_grad():
            bases = self.id_bases.cpu()  # [C, R, D]
            means = self.id_means.cpu()  # [C, D]
            C, R, D = bases.shape

            print(f"\n=== [Server Diagnostic] Subspace Orthogonality Check (C={C}, R={R}) ===")

            # ------------------------------------------------------
            # 指标 1: 子空间亲和度 (Subspace Affinity)
            # 衡量两个子空间的基向量是否平行。
            # 值范围 [0, 1]。0 = 完全正交(差异最大/最好)，1 = 完全重叠(最差)。
            # 计算方法：|| U_i @ U_j.T ||_F^2 / R
            # ------------------------------------------------------
            affinity_matrix = torch.zeros(C, C)
            for i in range(C):
                for j in range(C):
                    if i == j:
                        affinity_matrix[i, j] = 1.0
                        continue

                    # Basis i: [R, D], Basis j: [R, D]
                    # Interaction: [R, R]
                    interaction = torch.matmul(bases[i], bases[j].T)

                    # Frobenius Norm 归一化
                    norm = torch.norm(interaction, p='fro').item()
                    # 归一化到 [0, 1] (除以 sqrt(R) 因为最大 norm 是 sqrt(R))
                    score = norm / (np.sqrt(R) + 1e-8)
                    affinity_matrix[i, j] = score

            # 打印非对角线上的最大相似度（最坏情况）
            mask = ~torch.eye(C, dtype=bool)
            max_sim = affinity_matrix[mask].max().item()
            avg_sim = affinity_matrix[mask].mean().item()

            print(f"1. Subspace Overlap (Lower is better, 0=Orthogonal):")
            print(f"   - Max Overlap between distinct classes: {max_sim:.4f}")
            print(f"   - Avg Overlap between distinct classes: {avg_sim:.4f}")
            if max_sim > 0.5:
                print("   [WARNING] Some classes have very similar subspaces!")

            # ------------------------------------------------------
            # 指标 2: 相互重构测试 (Cross-Reconstruction)
            # 检查：用 Class A 的子空间重构 Class B 的均值，残差大不大？
            # 这里的残差是“相对残差” (Residual / Norm)。
            # 值接近 1.0 表示完全无法重构（差异大/好），接近 0.0 表示完美重构（差）。
            # ------------------------------------------------------
            cross_res_matrix = torch.zeros(C, C)
            for i in range(C): # 目标子空间 (Basis i)
                basis_i = bases[i] # [R, D]
                mu_i = means[i]    # [D]

                for j in range(C): # 测试样本 (Mean j)
                    if i == j: continue

                    # 待测试的向量：Class j 的均值
                    # 注意：我们要看的是 j 相对于 i 的子空间的残差
                    # 必须先针对 i 的中心进行去中心化
                    target_vec = means[j] - mu_i # [D]

                    # 投影 & 重构
                    # (D,) @ (D, R) -> (R,)
                    coeff = torch.matmul(target_vec, basis_i.T)
                    recon = torch.matmul(coeff, basis_i)

                    # 相对残差
                    res = torch.sum((target_vec - recon)**2)
                    orig_norm = torch.sum(target_vec**2) + 1e-8

                    relative_res = res / orig_norm
                    cross_res_matrix[i, j] = relative_res.item()

            min_res = cross_res_matrix[mask].min().item()
            avg_res = cross_res_matrix[mask].mean().item()

            print(f"2. Cross-Reconstruction Residual (Higher is better, Max=1.0):")
            print(f"   - Min Residual (Worst case pair): {min_res:.4f}")
            print(f"   - Avg Residual: {avg_res:.4f}")

            if min_res < 0.1:
                print(f"   [WARNING] Some classes can be reconstructed by others! (Min Res: {min_res:.4f})")
                # 找出是哪一对
                bad_idx = (cross_res_matrix + torch.eye(C)*10).argmin()
                row = bad_idx // C
                col = bad_idx % C
                print(f"   -> Worst Pair: Class {col.item()} reconstructs well on Class {row.item()}'s subspace.")

            print("=========================================================\n")

    def train(self, dataset, lr, metric, logger, unl_dataset=None):
        if 'fmatch' not in cfg['loss_mode']:
            data_loader = make_data_loader({'train': dataset}, 'server')['train']
            unl_loader = None
            unl_iter = None
            if unl_dataset is not None:
                unl_loader = make_data_loader({'train': unl_dataset}, 'global', shuffle={'train': True})['train']
                unl_iter = iter(unl_loader)
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model.parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            if cfg['server']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(data_loader) * float(cfg['local_epoch'][1])))
            else:
                num_batches = None
            for epoch in range(1, cfg['server']['num_epochs'] + 1):
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    input_size = input['data'].size(0)
                    input = to_device(input, cfg['device'])
                    # --- 新增：同步获取无标签数据 ---
                    if unl_iter is not None:
                        try:
                            input_unl = next(unl_iter)
                        except StopIteration:
                            # 如果无标签数据较少，遍历完了就重置迭代器
                            unl_iter = iter(unl_loader)
                            input_unl = next(unl_iter)

                        # 预处理无标签数据
                        input_unl = collate(input_unl)
                        input_unl = to_device(input_unl, cfg['device'])

                        # 将无标签数据存入 input 字典，key 为 'unlabeled'
                        # 这样模型内部可以通过 input['unlabeled'] 拿到数据
                        input['unlabeled'] = input_unl
                    # -------------------------------
                    if 'subspaces' in cfg['sel_method']:
                        input['valid_class_mask'] = self.persistent_class_mask
                        input['all_means'] = self.all_means.to(cfg['device'])
                        input['all_vars'] = self.all_vars.to(cfg['device'])
                    if 'affinity' in cfg['sel_method'] and ((self.affinitys is not None and len(self.affinitys) > 0) or 'edl' in cfg['loss_mode']):



                        # 2. 转到当前 Device 并存入 input
                        if 'edl' not in cfg['loss_mode']:
                            # 1. 随机选择一个客户端的矩阵
                            # 注意：affinitys 列表里存的是 tensor，可能在 CPU 上
                            selected_matrix = random.choice(self.affinitys)
                            # 如果不是 None，正常转到当前 Device
                            input['batch_affinity'] = selected_matrix.to(cfg['device'])
                        else:
                            # 如果是 None，生成全 1 的矩阵

                            affinity_mat = torch.ones(
                                cfg['seen_class_num'],
                                cfg['seen_class_num'],
                                device=cfg['device']
                            )

                            input['batch_affinity'] = affinity_mat.to(cfg['device'])


                    if cfg['task'] == 'OSR' and len(self.prototypes) != 0:

                        if False:
                            # 确定当前的 Batch Size
                            try:
                                B = input['data'].shape[0]
                            except KeyError:
                                B = 16

                            if len(self.prototypes) < 2:
                                # 如果不足两个元素，直接使用第一个元素，并复制 B 次
                                proto = self.prototypes[0].to(cfg['device'])
                                input['prototypes'] = proto.unsqueeze(0).repeat(B, 1)
                            else:
                                mixed_prototypes_list = []
                                # === 关键步骤：重复 B 次采样和混合 ===
                                for _ in range(B):
                                    # 1. 独立采样两个原型
                                    proto_a, proto_b = random.sample(self.prototypes, 2)

                                    # 2. 独立生成混合系数 lambda（因为 input['lam'] 是一个标量）
                                    # 注意：这里需要重新生成 lambda，而不是重复使用 input['lam']
                                    # 如果要保持和 input['lam'] 相同的生成逻辑，应使用该逻辑代码。
                                    # 这里我们使用简单的 torch.rand(1)
                                    lam_sample = self.beta.sample()[0]
                                    lam_sample = 1

                                    # 3. 将选取的 tensor 转移到正确的设备
                                    proto_a = proto_a.to(cfg['device'])
                                    proto_b = proto_b.to(cfg['device'])

                                    # 4. 计算混合后的单个 prototype [D] 并添加到列表中
                                    mixed_prototype = (lam_sample * proto_a + (1 - lam_sample) * proto_b).detach()
                                    mixed_prototypes_list.append(mixed_prototype.unsqueeze(0)) # 变为 [1, D]

                                # 5. 使用 torch.cat() 沿着 Batch 维度堆叠起来
                                input['prototypes'] = torch.cat(mixed_prototypes_list, dim=0)
                                input['ood_threshold'] = self.ood_threshold_mean
                                if 'subspaces' in cfg['sel_method']:
                                    input['ood_threshold'] = 1


                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation = metric.evaluate(['Loss', 'Accuracy'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break
        else:
            data_loader = make_data_loader({'train': dataset}, 'server')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model.make_sigma_parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            if cfg['server']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(data_loader) * float(cfg['local_epoch'][1])))
            else:
                num_batches = None
            for epoch in range(1, cfg['server']['num_epochs'] + 1):
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    input_size = input['data'].size(0)
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    for k, v in model.named_parameters():
                        parameter_type = k.split('.')[-1]
                        if 'weight' in parameter_type or 'bias' in parameter_type:
                            v.grad[(v.grad.size(0) // 2):] = 0
                    optimizer.step()
                    evaluation = metric.evaluate(['Loss', 'Accuracy'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break
        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        self.model_state_dict = save_model_state_dict(model.state_dict())
        if 'promax' in cfg['sel_method']:
            if 'promax' in cfg['sel_method']:

                with torch.no_grad():

                    C = cfg['seen_class_num']
                    class_features = [[] for _ in range(C)]

                    # 1. 将模型设置为评估模式
                    model.eval()

                    # 2. 遍历数据加载器，提取特征
                    # 注意：这里需要使用一个独立的 data_loader，或者重用主循环中的 data_loader
                    # 但需要确保 data_loader 在每次 epoch 开始时都会重置
                    for i, input in enumerate(data_loader):
                        input = collate(input)
                        input = to_device(input, cfg['device'])

                        output = model(input)

                        feature_batch = output['rep']
                        label_batch = input['target']

                        # 遍历 batch 中的每个样本
                        for j in range(feature_batch.size(0)):
                            label = label_batch[j].item()

                            if label < C:
                                # 收集特征到 CPU，并断开梯度
                                class_features[label].append(feature_batch[j].cpu())

                    # 3. 计算每个类别的原型（平均值）
                    prototype_list = []
                    for c in range(C):
                        if class_features[c]:
                            stacked_features = torch.stack(class_features[c], dim=0)
                            mean_prototype = torch.mean(stacked_features, dim=0)
                            prototype_list.append(mean_prototype)
                        else:
                            print(f"Warning: Class {c} has no collected features. Skipping.")

                    # 4. 存储最终原型到 self.kernels
                    if prototype_list:
                        # 堆叠所有原型 (C, D)，存储在 self.kernels 中，用于下一个 epoch 的 OSR 逻辑
                        self.kernels = torch.stack(prototype_list, dim=0)
                        print(f"Epoch {epoch}: Successfully computed and stored kernels. Shape: {self.kernels.shape}")
                    else:
                        print(f"Epoch {epoch}: Error: No prototypes were computed.")

        if 'subspaces' in cfg['sel_method'] and 'entropy' not in cfg['sel_method']:
            with torch.no_grad():
                C = cfg['seen_class_num']
                class_features = [[] for _ in range(C)]

                # 设定每个类保留的固定秩 (Target Rank per Class)
                # 例如：设为 1，则每个类只保留最强的一个主方向（类似原型线）
                # 设为 5，则每个类保留前5个主方向
                target_rank = cfg.get('rank_per_class', 8)

                model.eval()

                # --- 1. 数据收集 (保持不变) ---
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    input = to_device(input, cfg['device'])
                    output = model(input)
                    feature_batch = output['rep']
                    label_batch = input['target']
                    input['all_means'] = self.all_means.to(cfg['device'])
                    input['all_vars'] = self.all_vars.to(cfg['device'])

                    # 2. [新增] 传入覆盖数量，用于开关控制
                    # input['num_valid_classes'] = self.persistent_class_mask.sum()


                    for j in range(feature_batch.size(0)):
                        label = label_batch[j].item()
                        if label < C:
                            class_features[label].append(feature_batch[j].cpu())

                # --- 2. 计算固定秩的子空间基底 ---
                bases_list = []
                means_list = []
                vars_list = []  # <--- [新增] 存储方差

                for c in range(C):
                    if class_features[c]:
                        stacked_features = torch.stack(class_features[c], dim=0)
                        # stacked_features = F.normalize(stacked_features, p=2, dim=1)

                        # A. 计算并保存中心
                        mu = torch.mean(stacked_features, dim=0)
                        means_list.append(mu)

                        var = torch.var(stacked_features, dim=0, unbiased=True) + 1e-6
                        vars_list.append(var)

                        # B. 中心化
                        centered_features = stacked_features - mu

                        # C. 样本数量检查
                        # 如果样本数 N 小于目标秩 target_rank，我们只能提取 N 个基向量
                        # Vh 的形状是 [min(N, D), D]
                        N, D = centered_features.shape

                        if N < 2:
                            # 样本太少无法计算有效方差方向，填零或跳过
                            print(f"Warning: Class {c} sample size ({N}) too small for SVD.")
                            # 为了保持列表长度对齐，填入零矩阵 [target_rank, D]
                            bases_list.append(torch.zeros(target_rank, D))
                            continue

                        # D. SVD 分解
                        try:
                            # full_matrices=False, Vh shape: [min(N,D), D]
                            try:
                                # PyTorch >= 1.9.0 (推荐写法)
                                # 返回 U, S, Vh (Vh 是 V 的转置/共轭转置)
                                _, S, Vh = torch.linalg.svd(centered_features, full_matrices=False)
                            except AttributeError:
                                # PyTorch < 1.9.0 (老版本写法)
                                # torch.svd 返回 U, S, V (注意这里是 V，不是 Vh)
                                _, S, V = torch.svd(centered_features)
                                # 为了与 linalg.svd 的返回值 Vh 保持一致，需要对 V 进行转置
                                Vh = V.t()
                            # _, S, Vh = torch.linalg.svd(centered_features, full_matrices=False)

                            # E. 固定秩截断 (Fixed Rank Truncation)
                            # 取前 target_rank 行。如果 SVD 得到的向量不够（N < target_rank），就全取
                            available_rank = Vh.size(0)
                            actual_k = min(target_rank, available_rank)

                            # 提取基向量 [actual_k, D]
                            basis_k = Vh[:actual_k, :].clone()

                            # F. 补齐 (Padding) - 可选
                            # 为了让 self.id_bases 能堆叠成整齐的 Tensor [C, K, D]，
                            # 如果 actual_k < target_rank，我们可以用 0 补齐
                            if actual_k < target_rank:
                                padding = torch.zeros(target_rank - actual_k, D)
                                basis_k = torch.cat([basis_k, padding], dim=0)

                            bases_list.append(basis_k)

                        except RuntimeError as e:
                            print(f"SVD error class {c}: {e}")
                            bases_list.append(torch.zeros(target_rank, D))

                    else:
                        print(f"Warning: Class {c} has no features.")
                        # 补零占位
                        d_dim = feature_batch.size(1) if 'feature_batch' in locals() else 512
                        means_list.append(torch.zeros(d_dim))
                        bases_list.append(torch.zeros(target_rank, d_dim))
                        vars_list.append(torch.ones(d_dim)) # 方差默认为1防止除零

                # --- 3. 存储结果 ---
                if bases_list:
                    # 现在的 bases_list 每个元素形状都是 [target_rank, D]
                    # 所以我们可以直接 Stack 起来，形成一个规整的 Tensor
                    # Shape: [C, Rank, D] -> 例如 [10, 5, 512]
                    self.id_bases = torch.stack(bases_list, dim=0)
                    self.id_means = torch.stack(means_list, dim=0)
                    self.id_vars = torch.stack(vars_list, dim=0) # <--- [新增] 存储方差张量
                    self.check_subspace_orthogonality()


                    print(f"Epoch {epoch}: Computed fixed-rank subspaces. "
                        f"Bases: {self.id_bases.shape}, Means: {self.id_means.shape}")
                else:
                    print(f"Epoch {epoch}: Error: No subspaces computed.")



        return


class Client:
    def __init__(self, client_id, model, data_split):
        self.client_id = client_id
        self.data_split = data_split
        self.model_state_dict = save_model_state_dict(model.state_dict())
        if 'fmatch' in cfg['loss_mode']:
            optimizer = make_optimizer(model.make_phi_parameters(), 'local')
        else:
            optimizer = make_optimizer(model.parameters(), 'local')
        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        self.active = False
        self.beta = torch.distributions.beta.Beta(torch.tensor([cfg['alpha']]), torch.tensor([cfg['alpha']]))
        self.verbose = cfg['verbose']
        if cfg['task'] == 'OSR':
            self.prototype = None
            self.selected_ood_maxlength = max(8 * (cfg['seen_class_num']), cfg['client']['batch_size']['train'])
            self.selected_ood_maxlength = cfg.get('select_length',self.selected_ood_maxlength)
            self.selected_ood_update_length = 1
            self.selected_ood_count = 0
            self.selected_ood_scores = deque(maxlen=self.selected_ood_maxlength)
            self.selected_ood_labels = deque(maxlen=self.selected_ood_maxlength)
            self.selected_ood_images = deque(maxlen=self.selected_ood_maxlength)
            self.ood_threshold_mean = 0
            self.selected_id_images = deque(maxlen=self.selected_ood_maxlength)
            self.selected_id_labels = deque(maxlen=self.selected_ood_maxlength)
            if 'promax' in cfg['sel_method']:
                self.kernels = None
            if 'subspaces' in cfg['sel_method']:
                D = 128 if cfg['data_name'] not in ['TinyImageNet','MIOTCD'] else 256 # 特征维度
                C = cfg['seen_class_num']
                rank_id = cfg.get('rank_per_class', 8) # ID 秩
                rank_ood = cfg.get('rank_per_class', 8) # OOD 秩 (通常和ID设为一样，或者单独设)

                # 1. ID Bases: [C, Rank, D]
                self.id_bases = torch.empty((C, rank_id, D), device=cfg['device'])

                # 2. ID Means: [C, D]
                self.id_means = torch.zeros((C, D), device=cfg['device'])

                # 3. OOD Bases: [Rank, D]
                self.ood_bases = torch.empty((rank_ood, D), device=cfg['device'])

                # 4. OOD Means: [D]
                self.ood_means = torch.zeros((D,), device=cfg['device'])
                self.all_means = torch.zeros(C, D, device='cpu')
                self.all_vars = torch.zeros(C, D, device='cpu')
                self.class_counts = torch.zeros(C, device='cpu')
            if 'affinity' in cfg['sel_method']:
                self.affinity_matrix = None

    def make_hard_pseudo_label(self, soft_pseudo_label):
        max_p, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
        mask = max_p.ge(cfg['threshold'])
        if 'openmatch' in cfg['loss_mode'] and 'fedopenmatch' not in cfg['loss_mode']:
            mask = mask.fill_(True)
        return hard_pseudo_label, mask

    def make_dataset(self, dataset, metric, logger):
        if 'sup' in cfg['loss_mode'] or (('openmatch' in cfg['loss_mode'] or 'iomatch' in cfg['loss_mode'] or 'divcon' in cfg['loss_mode'] or 'bdmatch' in cfg['loss_mode']) and 'fedopenmatch' not in cfg['loss_mode']):
            return dataset
        elif 'fix' in cfg['loss_mode']:
            # with torch.no_grad():
            use_no_grad_flag = (cfg['sel_method'] != 'grad')
            loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            grads_per_sample = []
            with OptionalNoGrad(use_no_grad_flag):
                data_loader = make_data_loader({'train': dataset}, 'global', shuffle={'train': False})['train']
                model = eval('models.{}(track=True).to(cfg["device"])'.format(cfg['model_name']))
                model.load_state_dict(self.model_state_dict)
                model.train(False)
                output = []
                target = []
                out_reps = []
                edl_output = []

                if cfg['task'] == 'OSR' or cfg['seen_class_num']:
                    input_list = []
                for i, input in enumerate(data_loader):

                    input = collate(input)
                    input = to_device(input, cfg['device'])
                    target_i = copy.deepcopy(input['target'])
                    if cfg['task'] == 'OSR' or cfg['seen_class_num']:
                        input['target'].fill_(-1)
                        input_list.append(input)

                    output_ = model(input)
                    # --- WiseOpen 风格的梯度收集逻辑 ---
                    if cfg['sel_method'] == 'grad':
                        # 1. 计算伪标签 (使用当前模型的最高置信度)
                        pseudo_labels = output_['target'][:,:-1].argmax(dim=-1).detach()

                        # 2. 计算每个样本的损失 (reduction='none'，返回向量)
                        losses = loss_fn(output_['target'][:,:-1], pseudo_labels) # shape: (N,)

                        # 3. 遍历 batch 中的每个样本损失
                        for j in range(losses.size(0)):
                            model.zero_grad() # 清除模型之前的梯度

                            # 4. 对单个标量损失进行反向传播
                            # 必须使用 retain_graph=True，因为模型会在同一次前向传播的图上重复计算梯度
                            losses[j].backward(retain_graph=True)

                            # 5. 收集梯度标量化分数 (使用梯度范数作为标量)
                            # 收集模型所有参数的梯度范数，然后求和/取平均/L2范数等

                            # ----------------------------------------------------
                            # 关键步骤：将整个模型的梯度汇总成一个标量
                            # ----------------------------------------------------
                            # 提取所有参数的梯度张量，过滤掉 None，并计算它们的平方 L2 范数之和
                            grad_norm = sum(
                                torch.norm(param.grad.detach(), p=2).pow(2)
                                for param in model.parameters() if param.grad is not None
                            ).item()
                            grads_per_sample.append(grad_norm) # 存储梯度的平方范数

                    # ----------------------------------------------------
                    # 将每个样本的梯度append进grads_per_sample
                    output_i = output_['target']
                    # target_i = input['target']
                    output.append(output_i.cpu())
                    target.append(target_i.cpu())
                    if 'promax' in cfg['sel_method'] or 'subspaces' in cfg['sel_method']:
                        out_reps.append(output_['rep'])

                    if 'edl' in cfg['loss_mode']:
                        edl_output.append(output_['edl_output'].detach().cpu())
                output_, input_ = {}, {}
                output_['target'] = torch.cat(output, dim=0)
                if cfg['task'] == 'OSR':
                    output_['target'] = output_['target'][:,:-1]
                input_['target'] = torch.cat(target, dim=0)
                if 'promax' in cfg['sel_method'] or 'subspaces' in cfg['sel_method']:
                    output_['rep'] = torch.cat(out_reps, dim=0)
                if 'edl' in cfg['loss_mode']:
                    output_['edl'] = torch.cat(edl_output, dim=0)




                output_['target'] = F.softmax(output_['target'], dim=-1)
                new_target, mask = self.make_hard_pseudo_label(output_['target'])


                if cfg['task'] == 'OSR':
                # 1. 提取前面类别维度（排除最后一维OOD类）
                    id_probs = output_['target']  # shape: (N, C)，C=training_target_size

                    # 2. 计算每个样本的最大置信度（前面类别维度中的最大值）
                    max_id_probs, _ = torch.max(id_probs, dim=-1)  # shape: (N,)

                    # 3. 按最大置信度升序排序（取最低置信度样本）
                    _, conf_sorted_indices = torch.sort(max_id_probs)  # 默认升序（小→大）

                    # ------------------ 样本选择逻辑 ------------------

                    if 'entropy' in cfg['sel_method'] and 'edl' not in cfg['loss_mode']:
                        # ================= [新增: 统计 ID 类特征信息 (Subspaces 逻辑)] =================
                        if 'subspaces' in cfg['sel_method']:
                            feature_map = output_['rep']  # 提取特征表示 (N, D)
                            num_classes = cfg['seen_class_num']
                            # 获取硬伪标签
                            pseudo_labels = output_['target'].argmax(dim=-1).detach()
                            # 获取最大置信度，用于可选的过滤（目前按你提供的模板暂不强制过滤）
                            max_probs = output_['target'].max(dim=-1)[0]

                            for c in range(num_classes):
                                # 找到被预测为类别 c 的样本掩码
                                mask_c = (pseudo_labels == c)

                                # 只有样本数 > 1 才能计算方差
                                if mask_c.sum() > 1:
                                    feats_c = feature_map[mask_c].detach()

                                    # 计算均值和无偏方差
                                    mu_c = torch.mean(feats_c, dim=0).cpu()
                                    var_c = torch.var(feats_c, dim=0, unbiased=True).cpu()

                                    # 更新统计量，以便后续上传服务器

                                    self.all_means[c] = mu_c
                                    self.all_vars[c] = var_c
                                    self.class_counts[c] = mask_c.sum().cpu()
                        if 'affinity' in cfg['sel_method']:
                            with torch.no_grad():
                                num_seen = cfg['seen_class_num']

                                # 1. 获取所有数据的 ID 类 Logits
                                # output_['target'] 是全量数据 [N, Total_Classes]
                                # 我们截取 [N, Seen_Classes]
                                probs = output_['target'][:, :num_seen]


                                # 3. 一次性计算共现矩阵 M = P^T * P
                                # [C, N] @ [N, C] -> [C, C]
                                # 这里的数值代表：在整个数据集上，两个类被同时预测为高概率的程度
                                matrix = torch.mm(probs.t(), probs)

                                # 4. 消除对角线 (自己与自己的共现没有意义，设为0)
                                matrix.fill_diagonal_(0)

                                # 5. 存入 self (转到 CPU 方便后续通信)
                                # 因为是全量数据，直接赋值即可，不需要 +=
                                self.affinity_matrix = matrix.cpu()
                        # 4.a. 熵方法：计算信息熵 H(x) = -sum(p * log(p))
                        log_probs = torch.log(id_probs + 1e-12)
                        entropy = -torch.sum(id_probs * log_probs, dim=-1) # shape: (N,)

                        # 4.b. 熵方法：按熵降序排序（取最高熵样本）
                        _, entropy_sorted_indices = torch.sort(entropy, descending=True)  # 降序 (大 → 小)

                        # 4.c. 熵方法：截取前 N 个低置信度样本和前 N 个高熵样本的索引集合
                        conf_top_indices = conf_sorted_indices[:self.selected_ood_maxlength]
                        S_conf = set(conf_top_indices.tolist())

                        entropy_top_indices = entropy_sorted_indices[:self.selected_ood_maxlength]
                        S_entropy = set(entropy_top_indices.tolist())

                        # 4.d. 熵方法：计算交集 S_final
                        S_final = S_conf.intersection(S_entropy)
                        # final_indices = list(S_final)
                        final_indices = list(S_entropy)

                        # 4.e. 熵方法：补充策略 - 如果交集为空，则取置信度最低和熵最高的样本
                        if not final_indices:
                            # 取置信度最低的样本 (conf_sorted_indices[0])
                            idx_low_conf = conf_sorted_indices[0].item()

                            # 取熵最高的样本 (entropy_sorted_indices[0])
                            idx_high_entropy = entropy_sorted_indices[0].item()

                            if idx_low_conf == idx_high_entropy:
                                final_indices.append(idx_low_conf)
                            else:
                                final_indices.append(idx_low_conf)
                                final_indices.append(idx_high_entropy)

                            # 截断以确保不超过 max_length
                            final_indices = final_indices[:self.selected_ood_maxlength]
                    elif 'promax' in cfg['sel_method'] and self.kernels is not None: # <---- 新增 ProMax 逻辑

                        # 4.g. ProMax方法：假设特征和原型可用 (必须确保在实际代码中可用)
                        # feature_map 应该在前面步骤中从 output_ 中获取
                        # self.prototypes 应该是 (C, D) 形状，C为类别数，D为特征维度

                        # **注意：如果 output_['feature'] 和 self.prototypes 未定义，此处会报错**
                        feature_map = output_['rep'] # shape: (N, D)
                        kernels = self.kernels.to(cfg['device'])      # shape: (C, D)

                        # 计算每个样本到所有原型的距离（例如欧氏距离）
                        # (N, D) - (1, C, D) -> (N, C, D)
                        diff = feature_map.unsqueeze(1) - kernels.unsqueeze(0)
                        # (N, C, D) -> (N, C)
                        distances = torch.norm(diff, dim=-1)

                        # 4.h. ProMax方法：找到每个样本到所有类原型的最小距离
                        min_distances, _ = torch.min(distances, dim=-1) # shape: (N,)

                        # 4.i. ProMax方法：按最小距离降序排序（最小距离越大，越可能是OOD）
                        # 降序排序 (大 → 小)
                        _, distance_sorted_indices = torch.sort(min_distances, descending=True)

                        # 4.j. ProMax方法：截取前N个最小距离最大的样本
                        # final_indices = distance_sorted_indices[:self.selected_ood_maxlength].cpu()
                        # ------------------ 新增 ProMax + 置信度混合逻辑 ------------------
                        if 'fix' in cfg['sel_method']:
                            # 4.j. 混合方法：截取前 N 个最小距离最大的索引集合 (ProMax 候选)
                            dist_top_indices = distance_sorted_indices[:self.selected_ood_maxlength]
                            S_dist = set(dist_top_indices.tolist())

                            # 4.k. 混合方法：截取前 N 个最大置信度最低的索引集合 (置信度候选)
                            # conf_sorted_indices 已在前面步骤 (3) 中计算
                            conf_top_indices = conf_sorted_indices[:self.selected_ood_maxlength]
                            S_conf = set(conf_top_indices.tolist())

                            # 4.l. 混合方法：计算交集 S_final
                            S_final = S_dist.intersection(S_conf)
                            final_indices = list(S_final)

                            # 4.m. 混合方法：补充策略 - 如果交集为空，则取最小距离最大和置信度最低的样本
                            if not final_indices:
                                # 取最小距离最大的样本
                                idx_high_dist = distance_sorted_indices[0].item()

                                # 取置信度最低的样本 (conf_sorted_indices[0] 已经在前面步骤 (3) 中计算)
                                idx_low_conf = conf_sorted_indices[0].item()

                                if idx_high_dist == idx_low_conf:
                                    final_indices.append(idx_high_dist)
                                else:
                                    final_indices.append(idx_high_dist)
                                    final_indices.append(idx_low_conf)

                            # 截断以确保不超过 max_length
                            final_indices = final_indices[:self.selected_ood_maxlength]
                            final_indices = torch.tensor(final_indices).cpu()
                        elif 'entropy' in cfg['sel_method']:

                            # 4.j. 熵方法：计算信息熵 H(x) (需要 id_probs)
                            log_probs = torch.log(id_probs + 1e-12)
                            entropy = -torch.sum(id_probs * log_probs, dim=-1) # shape: (N,)

                            # 4.k. 熵方法：按熵降序排序（取最高熵样本）
                            _, entropy_sorted_indices = torch.sort(entropy, descending=True)  # 降序 (大 → 小)

                            # 4.l. 混合方法：截取前 N 个最小距离最大的索引集合 (ProMax 候选)
                            dist_top_indices = distance_sorted_indices[:self.selected_ood_maxlength]
                            S_dist = set(dist_top_indices.tolist())

                            # 4.m. 混合方法：截取前 N 个高熵样本的索引集合 (Entropy 候选)
                            entropy_top_indices = entropy_sorted_indices[:self.selected_ood_maxlength]
                            S_entropy = set(entropy_top_indices.tolist())

                            # 4.n. 混合方法：计算交集 S_final
                            S_final = S_dist.intersection(S_entropy)
                            final_indices = list(S_final)

                            # 4.o. 混合方法：补充策略 - 如果交集为空，则取最小距离最大和熵最高的样本
                            if not final_indices:
                                # 取最小距离最大的样本
                                idx_high_dist = distance_sorted_indices[0].item()

                                # 取熵最高的样本
                                idx_high_entropy = entropy_sorted_indices[0].item()

                                if idx_high_dist == idx_high_entropy:
                                    final_indices.append(idx_high_dist)
                                else:
                                    final_indices.append(idx_high_dist)
                                    final_indices.append(idx_high_entropy)

                                # 截断以确保不超过 max_length
                                final_indices = final_indices[:self.selected_ood_maxlength]
                                final_indices = torch.tensor(final_indices).cpu()

                            else:
                                final_indices = torch.tensor(final_indices).cpu()

                        else:
                            # 4.j. ProMax方法（纯）：截取前N个最小距离最大的样本
                            final_indices = distance_sorted_indices[:self.selected_ood_maxlength].cpu()
                    elif cfg['sel_method'] == 'grad':

                        # 4.g. 梯度方法：将收集到的梯度分数转换为张量
                        # 假设 grads_per_sample 存储了每个样本的梯度平方范数
                        grad_scores = torch.tensor(grads_per_sample, device=max_id_probs.device)

                        # 4.h. 梯度方法：按梯度分数降序排序
                        # 梯度分数越大，表示样本越不稳定/不确定，越可能是 OOD 候选
                        _, grad_sorted_indices = torch.sort(grad_scores, descending=True)  # 降序 (大 → 小)

                        # 4.i. 梯度方法：截取前N个最高梯度分数的样本索引
                        final_indices = grad_sorted_indices[:self.selected_ood_maxlength].cpu()
                        print(grad_scores[final_indices],len(grad_scores))
                    elif 'subspaces' in cfg['sel_method'] and getattr(self, 'id_bases', None) is not None and 'edl' not in cfg['loss_mode']:
                        # 4.p. Subspace方法：基于子空间重构残差筛选
                        # 逻辑：如果样本不能被任何已知类的子空间重构（残差大），则为 OOD

                        feature_map = output_['rep']  # shape: (N, D)
                        # feature_map = F.normalize(feature_map, p=2, dim=1) # 根据你的策略决定是否保留
                        N, D = feature_map.shape

                        # 初始化残差矩阵 (N, C)
                        # print(self.id_bases)
                        C = len(self.id_bases)
                        residuals = torch.zeros(N, C, device=feature_map.device)

                        # 遍历每个已知类，计算重构误差
                        for c in range(C):
                            basis = self.id_bases[c].to(feature_map.device)
                            mu = self.id_means[c].to(feature_map.device)
                            z_centered = feature_map - mu
                            coeffs = torch.matmul(z_centered, basis.T)
                            reconstruction = torch.matmul(coeffs, basis)

                            # 相对残差计算
                            res_c = torch.sum((z_centered - reconstruction) ** 2, dim=1) / (torch.sum(z_centered ** 2, dim=1) + 1e-8)
                            residuals[:, c] = res_c
                        # for c in range(num_classes):
                            # 找到属于类别 c 的样本掩码
                            # 逻辑：伪标签是 c  AND  置信度 > 阈值
                            pseudo_labels = output_['target'][:,:].argmax(dim=-1).detach()
                            mask_c = (pseudo_labels == c) #& (max_id_probs >= cfg['threshold'])

                            # 样本数量必须 > 1 才能计算方差
                            if mask_c.sum() > 1:
                                # 提取特征 [N_c, D]
                                feats_c = feature_map[mask_c].detach()

                                # 计算均值 [D]
                                mu_c = torch.mean(feats_c, dim=0).cpu()

                                # 计算方差 [D] (unbiased=True 表示样本方差 / N-1)
                                var_c = torch.var(feats_c, dim=0, unbiased=True).cpu()

                                self.all_means[c] = mu_c
                                self.all_vars[c] = var_c
                                self.class_counts[c] = mask_c.sum().cpu()
                            else:
                                # 如果该类样本不足，保持为0，或者可以处理为 None
                                # Server端在聚合时需要检查 class_counts > 0
                                pass
                        print('seen_class_counts:',self.class_counts)
                        # 1. [Subspace] 取最小残差并降序排序 (大 -> 小)
                        min_residuals, _ = torch.min(residuals, dim=1)
                        _, subspace_sorted_indices = torch.sort(min_residuals, descending=True)

                        # 2. [Confidence] 强制计算并升序排序 (小 -> 大, 越小越像 OOD)
                        _, conf_sorted_indices = torch.sort(max_id_probs, descending=False)

                        # 3. [Entropy] 强制计算并降序排序 (大 -> 小, 越大越像 OOD)
                        log_probs = torch.log(id_probs + 1e-12)
                        entropy = -torch.sum(id_probs * log_probs, dim=-1)
                        _, entropy_sorted_indices = torch.sort(entropy, descending=True)

                        # ========================= [强制可视化对比 & 交集分析] =========================

                        # A. 定义可视化函数
                        def visualize_distribution(indices, name, top_k=60):
                            """
                            打印 Top-K 样本分布，但计算 Precision 时使用实际的截断长度。
                            """
                            if indices is None or len(indices) == 0:
                                return

                            # 1. 实际截断长度 (您算法真正使用的长度)
                            real_cutoff = self.selected_ood_maxlength

                            # 2. 准备数据 (计算实际训练用的 Precision)
                            # 截取前 real_cutoff 个
                            selection_tensor = indices[:real_cutoff].to(input_['target'].device)
                            real_targets = input_['target'][selection_tensor]
                            seen_num = cfg['seen_class_num']

                            # 计算精度
                            is_ood_real = (real_targets >= seen_num).float()
                            real_prec = is_ood_real.mean().item() if len(is_ood_real) > 0 else 0

                            # 3. 准备可视化字符串 (只打印前 top_k 个字符)
                            viz_tensor = indices[:top_k].to(input_['target'].device)
                            viz_targets = input_['target'][viz_tensor]
                            is_ood_viz = (viz_targets >= seen_num).cpu().numpy()
                            # 使用 'X' 代表 ID (Wrong), 'O' 代表 OOD (Correct)
                            vis_str = "".join(["O" if x else "X" for x in is_ood_viz])

                            print(f"   > {name:<20}: {vis_str} (Real Prec @ {real_cutoff}: {real_prec:.1%})")

                        # B. 定义交集分析函数
                        def visualize_intersection(indices_list, name):
                            """
                            打印交集样本的分布和精度
                            """
                            if not indices_list:
                                print(f"   > {name:<20}: [Empty Intersection] (Count: 0)")
                                return

                            # 转换为 Tensor
                            idx_tensor = torch.tensor(indices_list, device=input_['target'].device)

                            # 获取真实标签
                            real_targets = input_['target'][idx_tensor]
                            seen_num = cfg['seen_class_num']
                            is_ood = (real_targets >= seen_num).cpu().numpy()

                            # 计算精度和数量
                            count = len(indices_list)
                            prec = is_ood.sum() / count if count > 0 else 0

                            # 生成字符串
                            vis_str = "".join(["O" if x else "X" for x in is_ood])

                            print(f"   > {name:<20}: {vis_str} (Count: {count}, Prec: {prec:.1%})")

                        print(f"\n[Screening Comparison] Top-60 sorted samples (O=OOD, X=ID):")
                        # 1. 单独打印三种方法的排序效果
                        visualize_distribution(subspace_sorted_indices, "Subspace (Residual)")
                        visualize_distribution(conf_sorted_indices,     "Fix (Confidence)")
                        visualize_distribution(entropy_sorted_indices,  "Entropy")

                        # 2. 计算交集 (在 selected_ood_maxlength 处截断)
                        cutoff = 100

                        # 转为集合 (Set)
                        set_sub = set(subspace_sorted_indices[:cutoff].tolist())
                        set_fix = set(conf_sorted_indices[:cutoff].tolist())
                        set_ent = set(entropy_sorted_indices[:cutoff].tolist())

                        # 计算交集列表
                        inter_sub_fix = list(set_sub.intersection(set_fix))
                        inter_sub_ent = list(set_sub.intersection(set_ent))

                        print("-" * 20 + " Intersection Analysis " + "-" * 20)
                        # 3. 打印交集效果
                        visualize_intersection(inter_sub_fix, "Inter (Sub & Fix)")
                        visualize_intersection(inter_sub_ent, "Inter (Sub & Ent)")
                        print("-" * 80)
                        # ==============================================================================

                        # ------------------ 下面是具体的筛选逻辑 (保持原有流程) ------------------

                        if 'fix' in cfg['sel_method']:
                            # 混合方法：Subspace (高残差) AND Confidence (低置信度)
                            dist_top_indices = subspace_sorted_indices[:self.selected_ood_maxlength]
                            S_dist = set(dist_top_indices.tolist())

                            conf_top_indices = conf_sorted_indices[:self.selected_ood_maxlength]
                            S_conf = set(conf_top_indices.tolist())

                            S_final = S_dist.intersection(S_conf)
                            final_indices = list(S_final)

                            # 补救策略
                            if not final_indices:
                                idx_high_dist = subspace_sorted_indices[0].item()
                                idx_low_conf = conf_sorted_indices[0].item()
                                if idx_high_dist == idx_low_conf:
                                    final_indices.append(idx_high_dist)
                                else:
                                    final_indices.append(idx_high_dist)
                                    final_indices.append(idx_low_conf)

                            final_indices = final_indices[:self.selected_ood_maxlength]
                            final_indices = torch.tensor(final_indices).cpu()

                        elif 'entropy' in cfg['sel_method']:
                            # 混合方法：Subspace (高残差) AND Entropy (高熵)
                            dist_top_indices = subspace_sorted_indices[:self.selected_ood_maxlength]
                            S_dist = set(dist_top_indices.tolist())

                            entropy_top_indices = entropy_sorted_indices[:self.selected_ood_maxlength]
                            S_entropy = set(entropy_top_indices.tolist())

                            S_final = S_dist.intersection(S_entropy)
                            final_indices = list(S_final)

                            # 补救策略
                            if not final_indices:
                                idx_high_res = subspace_sorted_indices[0].item()
                                idx_high_ent = entropy_sorted_indices[0].item()
                                if idx_high_res == idx_high_ent:
                                    final_indices.append(idx_high_res)
                                else:
                                    final_indices.append(idx_high_res)
                                    final_indices.append(idx_high_ent)

                            final_indices = final_indices[:self.selected_ood_maxlength]
                            final_indices = torch.tensor(final_indices).cpu()

                        else:
                            # 默认：纯 Subspace 方法
                            final_indices = subspace_sorted_indices[:self.selected_ood_maxlength].cpu()
                    elif 'edl' in cfg['loss_mode']:
                        evidence_alpha = output_['edl']
                        K = evidence_alpha.shape[1]
                        S = torch.sum(evidence_alpha, dim=-1)
                        uncertainty_u = (K / S).cpu().numpy() # 获取所有样本的不确定性分数
                        max_probs, pseudo_labels = torch.max(id_probs, dim=-1)

                        # 使用 gather 从 EDL 的 alpha 向量中提取对应预测类的证据值 alpha_k
                        predicted_evidence = evidence_alpha.gather(1, pseudo_labels.unsqueeze(1)).squeeze(1)

                        # 计算综合得分：Softmax 置信度 * 对应证据值
                        # 对于靠近第 4 类的 OOD，即使 max_probs 高，如果 predicted_evidence 被压制了，总分依然很低
                        conf_scores = (max_probs * predicted_evidence).detach().cpu().numpy()
                        # norm_evid = (evidence_alpha - 1.0) / evidence_alpha
                        # sum_norm_evid = torch.sum(norm_evid, dim=1, keepdim=True)
                        S_total = torch.sum(evidence_alpha, dim=1, keepdim=True)

                        # 2. 修改归一化方法为标准的 e/S (即信念 b_i)
                        norm_evid = (evidence_alpha - 1.0) / (S_total + 1e-8)
                        sum_norm_evid = torch.sum(norm_evid, dim=1, keepdim=True)

                        # 计算 Dissonance 公式：
                        # diss = sum_i [ (norm_evid_i * sum_{j!=i} (norm_evid_j * Bal(i,j))) / sum_{j!=i} norm_evid_j ]
                        # 这里使用简化实现：
                        diff_matrix = torch.abs(sum_norm_evid - norm_evid)
                        dissonance = torch.sum(norm_evid * (diff_matrix / (sum_norm_evid + 1e-8)), dim=1)
                        uncertainty_u = dissonance.detach().cpu().numpy()

                        # 使用 Otsu 算法动态寻找阈值
                        def find_otsu_threshold(u_scores):
                            u_min, u_max = u_scores.min(), u_scores.max()
                            bins = np.linspace(u_min, u_max, 256)
                            hist, _ = np.histogram(u_scores, bins=bins)

                            final_thresh = u_min
                            max_var = 0

                            for i in range(1, 256):
                                # 分割成两组：高不确定性(OOD候选)和低不确定性(ID候选)
                                w0 = np.sum(hist[:i]) # 组1权重
                                w1 = np.sum(hist[i:]) # 组2权重
                                if w0 == 0 or w1 == 0: continue

                                mu0 = np.mean(u_scores[u_scores <= bins[i]])
                                mu1 = np.mean(u_scores[u_scores > bins[i]])

                                # 计算类间方差
                                var_between = w0 * w1 * (mu0 - mu1) ** 2
                                if var_between > max_var:
                                    max_var = var_between
                                    final_thresh = bins[i]
                            return final_thresh

                        dynamic_threshold = find_otsu_threshold(uncertainty_u)
                        # dynamic_threshold = find_otsu_threshold(conf_scores)

                        # 筛选出超过动态阈值的样本索引
                        final_indices = torch.where(torch.from_numpy(uncertainty_u) > dynamic_threshold)[0]
                        conf_scores_torch = torch.from_numpy(conf_scores)
                        # final_indices = torch.where(conf_scores_torch < dynamic_threshold)[0]

                        # # 5. 安全限制与 Top-K 回退
                        # # 如果通过 Otsu 选出的 OOD 太多，我们只取“置信度最低”的前 N 个
                        # max_allowed = self.selected_ood_maxlength
                        # if len(final_indices) > max_allowed:
                        #     # largest=False 表示取分数最小的那些（即最不像 ID 的）
                        #     _, bot_idx = torch.topk(conf_scores_torch, max_allowed, largest=False)
                        #     final_indices = bot_idx

                        # 安全限制：防止选出太多导致崩溃，也可以不设
                        max_allowed = self.selected_ood_maxlength * 2
                        if len(final_indices) > max_allowed:
                            _, top_idx = torch.topk(torch.from_numpy(uncertainty_u), max_allowed)
                            final_indices = top_idx

                    else: # 默认置信度方法

                        # 4.f. 置信度方法：截取前N个低置信度样本
                        final_indices = conf_sorted_indices[:self.selected_ood_maxlength]
                    if cfg.get('global_epoch', 11)> 10 and 'match' not in cfg['loss_mode']:
                        all_data = torch.cat([b['data'] for b in input_list], dim=0)
                        # 如果有 'aug' 等其他键也要拼
                        all_aug = torch.cat([b['aug'] for b in input_list], dim=0) if 'aug' in input_list[0] else None
                        # 遍历排序后的索引，提取样本并填充队列
                        if 'randomselectclient' in cfg['loss_mode']:
                            # 获取总样本数量
                            total_samples = all_data.shape[0]
                            # 获取原本计划选择的长度 (即 self.selected_ood_maxlength)
                            target_length = len(final_indices)

                            # 生成随机排列的索引，并截取指定长度
                            # 注意：确保索引在 CPU 上，与原本的 final_indices 保持一致
                            final_indices = torch.randperm(total_samples)[:target_length].cpu()
                        for idx in final_indices:
                            # 提取单个样本（input_list是batch列表，通过列表推导式展平为样本列表）
                            # sample = {k: v[idx % len(v)] for batch in input_list for k, v in batch.items()}
                            sample = {'data': all_data[idx].cpu()} # 记得转回 CPU 存队列
                            sample['aug'] = all_aug[idx].cpu()

                            self.selected_ood_images.append(sample)
                            self.selected_ood_labels.append(cfg['seen_class_num'])  # OOD标签
                            self.selected_ood_scores.append(max_id_probs[idx].item())  # 存储ID类最大概率

                        ood_set = set(final_indices.tolist())
                        total_samples = all_data.shape[0]
                        candidate_id_indices = [i for i in range(total_samples) if i not in ood_set]
                        _, edl_preds = torch.max(evidence_alpha, dim=-1)  # shape: (N,)

                        valid_id_indices = []
                        for idx in candidate_id_indices:
                            # 核心条件：伪标签 mask 通过，且 EDL 分类头输出 == 伪标签
                            if mask[idx] and (edl_preds[idx] == new_target[idx]):
                                valid_id_indices.append(idx)

                        final_id_indices = []
                        target_length = len(final_indices)
                        target_id_length = target_length # 假设 ID 需要选和 OOD 一样多的数量

                        if len(valid_id_indices) >= target_id_length:
                            # 如果匹配的充足，优先挑选置信度 (max_id_probs) 最高的
                            valid_tensor = torch.tensor(valid_id_indices)
                            sorted_valid_idx = valid_tensor[torch.argsort(max_id_probs[valid_tensor], descending=True)]
                            final_id_indices = sorted_valid_idx[:target_id_length].tolist()
                        else:
                            # 如果匹配的不够，先全部收下
                            final_id_indices.extend(valid_id_indices)
                            shortage = target_id_length - len(valid_id_indices)

                            # 从剩下的、不满足匹配条件的候选者中找高 uncertainty (低置信度) 的凑数
                            remaining_candidates = list(set(candidate_id_indices) - set(valid_id_indices))
                            if len(remaining_candidates) > 0:
                                remaining_tensor = torch.tensor(remaining_candidates)

                                # 修改重点：使用 descending=True 保证排在前面的是概率大的样本（Uncertainty低）
                                sorted_remain_idx = remaining_tensor[torch.argsort(max_id_probs[remaining_tensor], descending=True)]

                                # 取剩余名额，并增加 min 保护防止剩余样本总数不足 shortage
                                take_num = min(shortage, len(sorted_remain_idx))
                                final_id_indices.extend(sorted_remain_idx[:take_num].tolist())

                        # ================= 3. 执行 ID 样本入队 =================
                        for idx in final_id_indices:
                            sample = {'data': all_data[idx].cpu()}
                            if all_aug is not None:
                                sample['aug'] = all_aug[idx].cpu()

                            self.selected_id_images.append(sample)
                            # 使用伪标签作为 ID 训练的目标
                            self.selected_id_labels.append(new_target[idx].item())
                        # mask[final_indices] = False
                output_['mask'] = mask
                evaluation = metric.evaluate(['PAccuracy', 'MAccuracy', 'LabelRatio'], input_, output_)
                if cfg['task'] == 'OSR' and 'edl' not in cfg['loss_mode']:
                    # # --- OOD 筛选准确率评估新增逻辑 ---
                    # # 1. 提取被选中 OOD 样本的预测输出
                    # # output_['target'] shape: (N, C_seen + 1)
                    # selected_ood_output = output_['target'][top_indices]

                    # # 2. 创建 OOD 样本的目标标签
                    # # OOD 样本的标签是 cfg['seen_class_num'] (即模型输出的最后一维)
                    # ood_label = cfg['seen_class_num']
                    # num_ood_selected = len(top_indices)
                    # selected_ood_target = torch.full((num_ood_selected,), ood_label,
                    #                                   dtype=torch.long, device=selected_ood_output.device)

                    # # 3. 计算 OOD 筛选准确率
                    # # 这里的 Accuracy 函数输入的是 output (softmax/logits) 和 target (long tensor)
                    # ood_selection_acc = Accuracy(selected_ood_output, selected_ood_target)
                    # evaluation['OODSelAccuracy'] = ood_selection_acc
                    # print(ood_selection_acc)
                    # 1. 提取被选中 OOD 样本的真实标签
                    # input_['target'] 包含所有无标签样本的真实标签
                    selected_samples_true_target = input_['target'][final_indices]

                    # 2. 判断真实标签是否属于 OOD 类别
                    # OOD 样本的真实标签 >= cfg['seen_class_num']
                    # 假设 ID 标签为 0 到 C_seen-1，OOD 标签为 C_seen 及以上
                    ood_label_start = cfg['seen_class_num']

                    # 这是一个布尔张量：True 表示该样本是真正的 OOD 样本
                    is_true_ood = (selected_samples_true_target >= ood_label_start)

                    # 3. 计算 OOD 筛选的精确度 (Precision)
                    # 精确度 = (真正被选中的 OOD 数量) / (被选中的总数)
                    true_ood_count = torch.sum(is_true_ood).float()
                    num_ood_selected = len(final_indices)
                    print('num_ood:',num_ood_selected)

                    if num_ood_selected > 0:
                        ood_selection_precision = (true_ood_count / num_ood_selected).item() * 100.0 # 转换为百分比
                    else:
                        # 如果没有样本被选中（尽管这在您的代码逻辑中不太可能）
                        ood_selection_precision = 0.0

                    # 4. 更新 evaluation 字典
                    # 保持 mask[final_indices] = False 逻辑不变
                    mask[final_indices] = False
                    print('indices num:',len(final_indices),len(mask))

                    evaluation = metric.evaluate(['PAccuracy', 'MAccuracy', 'LabelRatio'], input_, output_)

                    # 替换 OODSelAccuracy 为 OODSelPrecision
                    evaluation['OODSelPrecision'] = ood_selection_precision
                    print(f"OOD Selection Precision: {ood_selection_precision:.2f}%",len(final_indices))
                    # ========================= [新增: ID vs OOD 置信度对比] =========================
                    # 1. 获取所有样本的最大 Softmax 概率 (置信度)
                    # output_['target'] 在前面已经做过 F.softmax 了，直接取 max
                    all_probs = output_['target'] # Shape: [N, K+1]

                    # 注意：如果模型输出包含 K+1 类 (OOD类)，我们需要看两部分：
                    # A. 样本被预测为 ID 类的最大概率 (前 K 类 max)
                    # B. 样本被预测为任意类的最大概率 (全 max)
                    # 这里我们通常关注模型有多确信它是"已知类"，所以取前 seen_class_num 的 max 更有意义
                    # 但为了通用性，先取全局 max
                    all_max_conf, _ = torch.max(all_probs, dim=-1)

                    # 2. 获取真实标签掩码
                    all_true_targets = input_['target']
                    seen_num = cfg['seen_class_num']

                    true_id_mask = all_true_targets < seen_num      # 真实的 ID 样本
                    true_ood_mask = all_true_targets >= seen_num    # 真实的 OOD 样本

                    # 3. 计算并打印平均置信度
                    if true_id_mask.any():
                        avg_id_conf = all_max_conf[true_id_mask].mean().item()
                    else:
                        avg_id_conf = 0.0

                    if true_ood_mask.any():
                        avg_ood_conf = all_max_conf[true_ood_mask].mean().item()
                    else:
                        avg_ood_conf = 0.0

                    print(f"Confidence Stats (Softmax Max):")
                    print(f"   - True ID Samples Mean Conf : {avg_id_conf:.4f}")
                    print(f"   - True OOD Samples Mean Conf: {avg_ood_conf:.4f}")

                    if avg_ood_conf > avg_id_conf:
                        print("   [WARNING] OOD samples have HIGHER confidence than ID! (Model Overconfident on OOD)")
                    elif avg_id_conf - avg_ood_conf < 0.1:
                        print("   [WARNING] Confidence gap is small. Threshold screening might fail.")
                    else:
                        print("   [OK] ID confidence is significantly higher.")
                    # ==============================================================================
                    # ================= [新增高级分析: 基于置信度排序的最佳截断点] =================
                    # 目的: 模拟"如果只用置信度筛选，选多少个最好？"，并与你的子空间方法对比

                    if len(all_max_conf) > 0 and False:
                        # 1. 排序: OOD 理论上置信度低，所以按置信度升序 (Low -> High)
                        # sorted_conf: 排序后的置信度值
                        # sorted_idx : 对应的样本索引
                        sorted_conf, sorted_idx = torch.sort(all_max_conf, descending=False)

                        # 2. 获取排序后的真实标签 (是否为 OOD)
                        # true_ood_mask 已经在上面定义过: all_true_targets >= seen_num
                        sorted_is_ood = true_ood_mask[sorted_idx]

                        # 3. 计算累积指标 (Vectorized Calculation)
                        # cumsum 得到截止到当前位置，共有多少个真正的 OOD (True Positives)
                        cumulative_tp = torch.cumsum(sorted_is_ood.float(), dim=0)
                        # 当前位置的总样本数 (k)
                        # cumulative_total = torch.arange(1, len(sorted_is_ood) + 1, device=cfg['device'])
                        # 修正：让 cumulative_total 跟随 cumulative_tp 所在的设备（通常是 CPU）
                        cumulative_total = torch.arange(1, len(sorted_is_ood) + 1, device=cumulative_tp.device)

                        # Precision @ k
                        curr_prec_curve = cumulative_tp / cumulative_total
                        # Recall @ k
                        total_real_ood = true_ood_mask.sum().float()
                        if total_real_ood > 0:
                            curr_rec_curve = cumulative_tp / total_real_ood
                        else:
                            curr_rec_curve = torch.zeros_like(curr_prec_curve)

                        # 4. 计算 F1-Score 曲线，找到峰值 (最合适的截断点)
                        f1_curve = 2 * (curr_prec_curve * curr_rec_curve) / (curr_prec_curve + curr_rec_curve + 1e-8)
                        best_idx = torch.argmax(f1_curve).item()
                        best_k = best_idx + 1 # 最佳样本数量

                        print(f"\n[Analysis] Optimal Truncation based purely on Confidence (Low->High):")

                        # --- A. 与当前 Subspace 方法的横向对比 ---
                        curr_len = len(final_indices) # 你当前选了多少个
                        if curr_len > 0 and curr_len <= len(curr_prec_curve):
                            # 如果单纯用置信度选同样数量的样本，精度是多少？
                            conf_prec_at_curr = curr_prec_curve[curr_len-1].item() * 100
                            print(f"   A. Comparison at current selection length (K={curr_len}):")
                            print(f"      -> Subspace Algo Precision : {ood_selection_precision:.2f}% (Your Method)")
                            print(f"      -> Confidence Sort Precision: {conf_prec_at_curr:.2f}% (Baseline)")
                            if ood_selection_precision > conf_prec_at_curr:
                                print("      => RESULT: Subspace works BETTER than simple confidence!")
                            else:
                                print("      => RESULT: Subspace is WORSE or EQUAL. Check feature norms.")

                        # --- B. F1 最优截断点 (综合性能最好) ---
                        best_conf_threshold = sorted_conf[best_idx].item()
                        print(f"   B. Optimal F1 Cutoff (Most Balanced):")
                        print(f"      -> Best Length (K) : {best_k}")
                        print(f"      -> Precision       : {curr_prec_curve[best_idx].item()*100:.2f}%")
                        print(f"      -> Recall          : {curr_rec_curve[best_idx].item()*100:.2f}%")
                        print(f"      -> Conf Threshold  : < {best_conf_threshold:.4f}")

                        # --- C. 高精度截断点 (如果只想选"非常稳"的 OOD) ---
                        # 寻找 Precision >= 90% 的最大长度
                        high_prec_mask = curr_prec_curve >= 0.90
                        if high_prec_mask.any():
                            # 找到满足条件的最后一个索引
                            max_len_90 = torch.max(torch.where(high_prec_mask)[0]).item() + 1
                            print(f"   C. Max Length with Precision >= 90%:")
                            print(f"      -> Length (K)      : {max_len_90}")
                            print(f"      -> Conf Threshold  : < {sorted_conf[max_len_90-1].item():.4f}")
                        else:
                            print(f"   C. Max Length with Precision >= 90%: None (Precision never reached 90%)")

                        print("==========================================================================")

                logger.append(evaluation, 'train', n=len(input_['target']))
                if torch.any(mask):
                    fix_dataset = copy.deepcopy(dataset)
                    fix_dataset.target = new_target.tolist()
                    mask = mask.tolist()
                    fix_dataset.data = list(compress(fix_dataset.data, mask))
                    fix_dataset.target = list(compress(fix_dataset.target, mask))
                    fix_dataset.other = {'id': list(range(len(fix_dataset.data)))}
                    if 'mix' in cfg['loss_mode']:
                        mix_dataset = copy.deepcopy(dataset)
                        mix_dataset.target = new_target.tolist()
                        mix_dataset = MixDataset(len(fix_dataset), mix_dataset)
                    else:
                        mix_dataset = None
                    return fix_dataset, mix_dataset
                else:
                    return None
        else:
            raise ValueError('Not valid client loss mode')

    def train(self, dataset, lr, metric, logger):
        if cfg['loss_mode'] == 'sup':
            data_loader = make_data_loader({'train': dataset}, 'client')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict, strict=False)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            if self.optimizer_state_dict is not None:
                optimizer = make_optimizer(model.parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            if cfg['client']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(data_loader) * float(cfg['local_epoch'][0])))
            else:
                num_batches = None
            for epoch in range(1, cfg['client']['num_epochs'] + 1):
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    input_size = input['data'].size(0)
                    input['loss_mode'] = cfg['loss_mode']
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation = metric.evaluate(metric.metric_name['train'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break
        elif 'fix' in cfg['loss_mode'] and 'mix' not in cfg['loss_mode'] and 'batch' not in cfg[
            'loss_mode'] and 'frgd' not in cfg['loss_mode'] and 'fmatch' not in cfg['loss_mode']:
            if not isinstance(dataset, (list, tuple)) or len(dataset) < 1:
                fix_dataset = dataset
            else:
                fix_dataset, _ = dataset
            fix_data_loader = make_data_loader({'train': fix_dataset}, 'client')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict, strict=False)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model.parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            if cfg['client']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(fix_data_loader) * float(cfg['local_epoch'][0])))
            else:
                num_batches = None
            for epoch in range(1, cfg['client']['num_epochs'] + 1):
                for i, input in enumerate(fix_data_loader):
                    input = collate(input)
                    input_size = input['data'].size(0)
                    input['loss_mode'] = cfg['loss_mode']
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation = metric.evaluate(['Loss', 'Accuracy'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break
        elif 'fix' in cfg['loss_mode'] and 'mix' in cfg['loss_mode'] and 'batch' not in cfg[
            'loss_mode'] and 'frgd' not in cfg['loss_mode'] and 'fmatch' not in cfg['loss_mode']:
            fix_dataset, mix_dataset = dataset
            fix_data_loader = make_data_loader({'train': fix_dataset}, 'client')['train']
            mix_data_loader = make_data_loader({'train': mix_dataset}, 'client')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict, strict=False)
            optimizer = make_optimizer(model.parameters(), 'local')
            if self.optimizer_state_dict is not None:
                self.optimizer_state_dict['param_groups'][0]['lr'] = lr
                optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            if cfg['client']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(fix_data_loader) * float(cfg['local_epoch'][0])))
            else:
                num_batches = None
            for epoch in range(1, cfg['client']['num_epochs'] + 1):
                for i, (fix_input, mix_input) in enumerate(zip(fix_data_loader, mix_data_loader)):
                    input = {'data': fix_input['data'], 'target': fix_input['target'], 'aug': fix_input['aug'],
                             'mix_data': mix_input['data'], 'mix_target': mix_input['target']}
                    input = collate(input)
                    input_size = input['data'].size(0)
                    input['lam'] = self.beta.sample()[0]
                    input['mix_data'] = (input['lam'] * input['data'] + (1 - input['lam']) * input['mix_data']).detach()

                    if cfg['open_close_ratio']:
                        input['mix_aug'] = torch.stack(mix_input['aug'], 0)
                        input['mix_aug'] = (input['lam'] * input['aug'] + (1 - input['lam']) * input['mix_aug']).detach()
                    input['mix_target'] = torch.stack([input['target'], input['mix_target']], dim=-1)
                    input['loss_mode'] = cfg['loss_mode']
                    input = to_device(input, cfg['device'])
                    if cfg['task'] == 'OSR':
                        if len(self.selected_ood_images) > 0:
                            input['selected_ood_images'] = self.selected_ood_images
                            input['selected_ood_labels'] = self.selected_ood_labels
                            input['selected_ood_scores'] = self.selected_ood_scores
                        if len(self.selected_id_images) > 0:
                            input['selected_id_images'] = self.selected_id_images
                            input['selected_id_labels'] = self.selected_id_labels



                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation = metric.evaluate(['Loss', 'Accuracy'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break
        elif 'batch' in cfg['loss_mode'] or 'frgd' in cfg['loss_mode'] or 'fmatch' in cfg['loss_mode']:
            data_loader = make_data_loader({'train': dataset}, 'client')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict, strict=False)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            if 'fmatch' in cfg['loss_mode']:
                optimizer = make_optimizer(model.make_phi_parameters(), 'local')
            else:
                optimizer = make_optimizer(model.parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            if cfg['client']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(data_loader) * float(cfg['local_epoch'][0])))
            else:
                num_batches = None
            for epoch in range(1, cfg['client']['num_epochs'] + 1):
                for i, input in enumerate(data_loader):
                    with torch.no_grad():
                        model.train(False)
                        input_ = collate(input)
                        input_ = to_device(input_, cfg['device'])
                        output_ = model(input_)
                        output_i = output_['target']
                        output_['target'] = F.softmax(output_i, dim=-1)
                        new_target, mask = self.make_hard_pseudo_label(output_['target'])
                        output_['mask'] = mask
                        evaluation = metric.evaluate(['PAccuracy', 'MAccuracy', 'LabelRatio'], input_, output_)
                        logger.append(evaluation, 'train', n=len(input_['target']))
                    if torch.all(~mask):
                        continue
                    model.train(True)
                    input = {'data': input['data'][mask], 'aug': input['aug'][mask], 'target': new_target[mask]}
                    input = to_device(input, cfg['device'])
                    input_size = input['data'].size(0)
                    input['loss_mode'] = 'fix'
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation = metric.evaluate(['Loss', 'Accuracy'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break
        else:
            raise ValueError('Not valid client loss mode')
        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        self.model_state_dict = save_model_state_dict(model.state_dict())
        if cfg['task'] == 'OSR':
            self.prototype = None
        if hasattr(self, 'selected_ood_images') and self.selected_ood_images is not None and len(self.selected_ood_images) != 0:
            if 'subspaces' not in cfg['sel_method'] and False:


                data_tensors = [item['data'].unsqueeze(0).to(cfg['device']) for item in self.selected_ood_images]
                # 步骤 2: 使用 torch.cat() 在第一维 (dim=0) 上将这些张量拼接起来
                # 结果张量的第一维将是所有单独张量第一维之和 (16 + 32 + 8 = 56)
                collected_data_tensor = torch.cat(data_tensors, dim=0)
                input1 = {'data':collected_data_tensor,'target':torch.tensor(-1, device=cfg['device'])}
                out=model(input1)
                dp_beta = cfg.get('dp_beta', 0.1)
                if cfg['protection'] == 1:
                    local_ood_prototype = torch.mean(out['rep'], dim=0).detach()
                    # 4a. 计算 OOD 样本特征的方差向量 v
                    # dim=0 (batch_size) 上求方差，得到 v (feature_dim,)
                    # 设置 unbiased=False 对应样本方差 (N-1) vs 总体方差 (N) 的选择，
                    # 在 DP 中通常使用 N 作为分母。
                    ood_feature_variance = torch.var(out['rep'], dim=0, unbiased=False).detach()

                    # 4b. 计算高斯噪声的标准差向量 sigma = sqrt(beta * v)
                    # noise_std 形状: (feature_dim,)
                    noise_std = torch.sqrt(dp_beta * ood_feature_variance)

                    # 4c. 生成各向异性噪声 xi ~ N(0, diag(beta * v))
                    # 噪声 xi 形状: (feature_dim,)
                    xi = torch.randn_like(local_ood_prototype) * noise_std

                    # 4d. 计算加噪后的 OOD 原型 (p_hat_ood)
                    self.prototype = (local_ood_prototype + xi).to('cpu')
                elif cfg['protection'] == 2:
                    local_ood_prototype = torch.mean(out['rep'], dim=0).detach()

                    # 4a. 确定各向同性标准差 (标量)
                    # 在所有维度上使用相同的标准差 dp_sigma_iso

                    # 4b. 生成各向同性噪声 xi ~ N(0, sigma_iso^2 * I)
                    # 噪声 xi 形状: (feature_dim,)
                    dp_sigma_iso = cfg.get('dp_sigma_iso', 0.01)
                    xi = torch.randn_like(local_ood_prototype) * dp_sigma_iso

                    # 4c. 计算加噪后的 OOD 原型
                    self.prototype = (local_ood_prototype + xi).to('cpu')
                else:
                    self.prototype = torch.mean(out['rep'],dim=0).detach().to('cpu')
                # out['target'] 是输出的概率分布 (假设是 Logits)
                # 1. 确定已知类数量 K
                num_known_classes = cfg['seen_class_num']
                # 2. 提取前 K 类的 Logits
                # 形状: (batch_size, num_known_classes)
                known_logits = out['target'][:, :num_known_classes]
                # 3. 对已知类部分进行 Softmax
                known_probs = F.softmax(known_logits, dim=1)
                # 4. 找出已知类中的最大概率 (MSP)
                # 形状: (batch_size,)
                max_known_probs = torch.max(known_probs, dim=1)[0]
                # 5. 对 MSP 值求平均作为动态阈值
                # 使用 .item() 转换成 Python 浮点数，方便在 cfg 中作为常量使用
                self.ood_threshold_mean = max_known_probs.mean().item()

            elif 'subspaces' in cfg['sel_method'] and False:
                # 检查是否有筛选出的 OOD 样本
                if hasattr(self, 'selected_ood_images') and self.selected_ood_images is not None and len(self.selected_ood_images) > 0:
                    with torch.no_grad():
                        # 0. 获取目标秩 (Target Rank)
                        # 设为 8 (默认)，则每个客户端将上传 8 个 OOD 主方向
                        target_rank = cfg.get('rank_per_class', 8)

                        # 1. 准备数据 (将 List of Tensors 拼成 Batch)
                        data_tensors = [item['data'].unsqueeze(0).to(cfg['device']) for item in self.selected_ood_images]
                        collected_data_tensor = torch.cat(data_tensors, dim=0) # [N, C, H, W]

                        # 构造输入 (Target设为-1或忽略)
                        input_ood = {'data': collected_data_tensor, 'target': torch.full((len(data_tensors),), -1, device=cfg['device'])}

                        # 2. 通过模型提取特征
                        model.eval()
                        out_ood = model(input_ood)
                        ood_rep = out_ood['rep'] # [N, Feature_Dim]

                        # 3. 计算均值 (Centroid)
                        ood_mean = torch.mean(ood_rep, dim=0)

                        # 4. SVD 计算基底
                        # 中心化
                        ood_centered = ood_rep - ood_mean

                        # 只有样本数 > 1 才能计算有效的方差方向
                        if ood_centered.size(0) > 1:
                            try:
                                # torch.linalg.svd: U, S, Vh
                                # Vh 的每一行即为特征向量 (基底)
                                try:
                                    # PyTorch >= 1.9.0 (推荐写法)
                                    # 返回 U, S, Vh (Vh 是 V 的转置/共轭转置)
                                    _, S, Vh = torch.linalg.svd(ood_centered, full_matrices=False)
                                except AttributeError:
                                    # PyTorch < 1.9.0 (老版本写法)
                                    # torch.svd 返回 U, S, V (注意这里是 V，不是 Vh)
                                    _, S, V = torch.svd(ood_centered)
                                    # 为了与 linalg.svd 的返回值 Vh 保持一致，需要对 V 进行转置
                                    Vh = V.t()
                                # _, S, Vh = torch.linalg.svd(ood_centered, full_matrices=False)

                                # --- 修改：使用固定 Rank 截断 ---
                                # Vh shape: [min(N, D), D]
                                available_rank = Vh.size(0)
                                actual_k = min(target_rank, available_rank)

                                # 提取前 actual_k 个基向量
                                ood_basis = Vh[:actual_k, :].clone()

                                # --- 补齐 (Padding) ---
                                # 如果实际计算出的秩小于目标秩 (例如只有 3 个样本，但 target_rank=8)
                                # 用全零向量补齐，保证上传形状统一为 [target_rank, D]
                                if actual_k < target_rank:
                                    padding = torch.zeros(target_rank - actual_k, ood_rep.size(1), device=cfg['device'])
                                    ood_basis = torch.cat([ood_basis, padding], dim=0)

                            except RuntimeError as e:
                                print(f"OOD SVD computation failed: {e}")
                                # 失败则给一个全零基底占位 [target_rank, D]
                                ood_basis = torch.zeros(target_rank, ood_rep.size(1), device=cfg['device'])
                        else:
                            # 样本太少 (N=1)，无法形成子空间，给全零
                            # print("Warning: Only 1 OOD sample selected.")
                            ood_basis = torch.zeros(target_rank, ood_rep.size(1), device=cfg['device'])

                        # 5. 存储结果 (转至 CPU 以备上传)
                        # self.ood_bases: [target_rank, D]
                        # self.ood_means: [D]
                        self.ood_bases = ood_basis.cpu()
                        self.ood_means = ood_mean.cpu()

                        # print(f"Computed OOD Subspace -> Means: {self.ood_means.shape}, Bases: {self.ood_bases.shape}")
                else:
                    # 如果没有筛选出 OOD 样本，设置为空或 None
                    self.ood_bases = None
                    self.ood_means = None

            # 清空队列
            self.selected_ood_scores = deque(maxlen=self.selected_ood_maxlength)
            self.selected_ood_labels = deque(maxlen=self.selected_ood_maxlength)
            self.selected_ood_images = deque(maxlen=self.selected_ood_maxlength)
            self.selected_ood_scores = deque(maxlen=self.selected_ood_maxlength)
            self.selected_ood_labels = deque(maxlen=self.selected_ood_maxlength)
            self.selected_ood_images = deque(maxlen=self.selected_ood_maxlength)
            self.selected_id_images = deque(maxlen=self.selected_ood_maxlength)
            self.selected_id_labels = deque(maxlen=self.selected_ood_maxlength)

        return


def save_model_state_dict(model_state_dict):
    return {k: v.cpu() for k, v in model_state_dict.items()}


def save_optimizer_state_dict(optimizer_state_dict):
    optimizer_state_dict_ = {}
    for k, v in optimizer_state_dict.items():
        if k == 'state':
            optimizer_state_dict_[k] = to_device(optimizer_state_dict[k], 'cpu')
        else:
            optimizer_state_dict_[k] = copy.deepcopy(optimizer_state_dict[k])
    return optimizer_state_dict_
