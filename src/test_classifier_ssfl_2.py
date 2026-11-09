import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import models
from config import cfg, process_args
from data import fetch_dataset, make_data_loader, make_batchnorm_stats
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, resume, collate
from logger import make_logger
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsRegressor # [新增] 用于映射新样本到已有的 t-SNE 空间
import sys

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


# ================= 虚拟 OOD 样本生成函数 =================
def generate_batch_based_ood_1(reps, targets, affinity_matrix, device, mix_ratio=1.0, fill_diagonal=False):
    unique_classes = torch.unique(targets)

    if len(unique_classes) < 2:
        return torch.randn_like(reps)[:len(reps)]

    class_to_indices = {}
    for c in unique_classes:
        class_to_indices[c.item()] = (targets == c).nonzero(as_tuple=True)[0]

    mask = torch.zeros_like(affinity_matrix, dtype=torch.bool)
    try:
        grid_x, grid_y = torch.meshgrid(unique_classes, unique_classes, indexing='ij')
    except TypeError:
        grid_x, grid_y = torch.meshgrid(unique_classes, unique_classes)
    mask[grid_x, grid_y] = True
    if not fill_diagonal:
        mask.fill_diagonal_(False)

    masked_affinity = affinity_matrix * mask.float()
    total_affinity = masked_affinity.sum()
    if total_affinity > 0:
        probs = masked_affinity.view(-1) / total_affinity
    else:
        probs = mask.view(-1).float() / mask.sum()

    num_gen = reps.size(0)

    pair_indices = torch.multinomial(probs, num_gen, replacement=True)
    num_classes = affinity_matrix.shape[0]

    class_a_list = pair_indices // num_classes
    class_b_list = pair_indices % num_classes

    # lam = torch.distributions.Beta(2.0, 2.0).sample((num_gen, 1)).to(device)
    lam = (torch.rand((num_gen, 1), device=device) * 0.2 + 0.4)

    feats_a = []
    feats_b =[]

    for i in range(num_gen):
        c_a = class_a_list[i].item()
        c_b = class_b_list[i].item()

        idx_tensor_a = class_to_indices[c_a][torch.randint(len(class_to_indices[c_a]), (1,))]
        idx_a = idx_tensor_a.item()

        idx_tensor_b = class_to_indices[c_b][torch.randint(len(class_to_indices[c_b]), (1,))]
        idx_b = idx_tensor_b.item()

        feats_a.append(reps[idx_a])
        feats_b.append(reps[idx_b])

    feats_a = torch.stack(feats_a).to(device)
    feats_b = torch.stack(feats_b).to(device)

    mixed_features = lam * feats_a + (1 - lam) * feats_b
    return mixed_features
# ===============================================================
def generate_batch_based_ood_2(reps, targets, affinity_matrix, device, real_ood_reps=None, mix_ratio=1.0, fill_diagonal=False):
    unique_classes = torch.unique(targets)

    if len(unique_classes) < 2:
        return torch.randn_like(reps)[:len(reps)]

    class_to_indices = {}
    for c in unique_classes:
        class_to_indices[c.item()] = (targets == c).nonzero(as_tuple=True)[0]

    mask = torch.zeros_like(affinity_matrix, dtype=torch.bool)
    try:
        grid_x, grid_y = torch.meshgrid(unique_classes, unique_classes, indexing='ij')
    except TypeError:
        grid_x, grid_y = torch.meshgrid(unique_classes, unique_classes)
    mask[grid_x, grid_y] = True
    if not fill_diagonal:
        mask.fill_diagonal_(False)

    masked_affinity = affinity_matrix * mask.float()
    total_affinity = masked_affinity.sum()
    if total_affinity > 0:
        probs = masked_affinity.view(-1) / total_affinity
    else:
        probs = mask.view(-1).float() / mask.sum()

    num_gen = reps.size(0)

    pair_indices = torch.multinomial(probs, num_gen, replacement=True)
    num_classes = affinity_matrix.shape[0]

    class_a_list = pair_indices // num_classes
    class_b_list = pair_indices % num_classes

    # 1. 基础采样逻辑 (保持高效的类别索引)
    class_to_indices = {c.item(): (targets == c).nonzero(as_tuple=True)[0] for c in unique_classes}
    num_gen = reps.size(0)

    # 2. 这里的 lam 依然保持在 [0.3, 0.7]，不让它太靠近任何一个 ID 类
    lam = (torch.rand((num_gen, 1), device=device) * 0.4 + 0.3)

    # 快速获取配对特征
    # (为了简洁，这里省略了 probs 计算逻辑，假设已根据 affinity_matrix 选好 class_a/b)
    # ... [此处对接你原有的 pair_indices 逻辑] ...

    feats_a = torch.stack([reps[class_to_indices[ca.item()][torch.randint(len(class_to_indices[ca.item()]), (1,))]] for ca in class_a_list]).to(device).squeeze()
    feats_b = torch.stack([reps[class_to_indices[cb.item()][torch.randint(len(class_to_indices[cb.item()]), (1,))]] for cb in class_b_list]).to(device).squeeze()

    # 3. 基础混合：得到处于 ID 边界的中间点
    mixed_features = lam * feats_a + (1 - lam) * feats_b

    # --- 核心改进：向 OOD 区域“偏移” ---

    if real_ood_reps is not None and len(real_ood_reps) > 0:
        # 【方案 A】：已知真实 OOD，向 OOD 中心拉近
        ood_center = torch.mean(real_ood_reps, dim=0, keepdim=True)
        # 偏移权重 alpha=0.2 意味着向 OOD 靠近 20%，既有 OOD 特性又不至于完全重合
        alpha = 0.2
        mixed_features = (1 - alpha) * mixed_features + alpha * ood_center
    else:
        # 【方案 B】：无真实 OOD，向 ID 流形的外部推离
        id_center = torch.mean(reps, dim=0, keepdim=True)
        # 计算从 ID 中心指向合成点的向量
        direction = mixed_features - id_center
        # 稍微外推 10% 的距离，让它离开 ID 高密区
        mixed_features = mixed_features + 0.1 * direction

    # 4. 最后的扰动：让样本在目标区域“散开”，而不是一个死点
    reps_std = torch.std(reps, dim=0, keepdim=True)
    noise = torch.randn_like(mixed_features) * reps_std * 0.15 # 适度噪声
    mixed_features = mixed_features + noise

    return mixed_features
def generate_batch_based_ood(reps, targets, affinity_matrix, device, real_ood_reps=None):
    """
    基于当前 Batch 的特征和选定的 Affinity Matrix 生成 OOD。
    增加了 3 个小妙招提升合成质量：边界截断、加噪膨胀、特征范数对齐。
    """
    unique_classes = torch.unique(targets)
    if len(unique_classes) < 2:
        return torch.randn_like(reps)[:len(reps)]

    class_to_indices = {}
    for c in unique_classes:
        class_to_indices[c.item()] = (targets == c).nonzero(as_tuple=True)[0]

    mask = torch.zeros_like(affinity_matrix, dtype=torch.bool)
    try:
        grid_x, grid_y = torch.meshgrid(unique_classes, unique_classes, indexing='ij')
    except TypeError:
        grid_x, grid_y = torch.meshgrid(unique_classes, unique_classes)
    mask[grid_x, grid_y] = True
    mask.fill_diagonal_(False)

    masked_affinity = affinity_matrix * mask.float()
    total_affinity = masked_affinity.sum()
    if total_affinity > 0:
        probs = masked_affinity.view(-1) / total_affinity
    else:
        probs = mask.view(-1).float() / mask.sum()

    num_gen = reps.size(0)
    pair_indices = torch.multinomial(probs, num_gen, replacement=True)
    num_classes = affinity_matrix.shape[0]

    class_a_list = pair_indices // num_classes
    class_b_list = pair_indices % num_classes

    # 【妙招 1】：截断的 Mixup。不使用 Beta(2,2)，改用[0.2, 0.8] 的均匀分布。
    # 强制合成的点待在两个类别中间，绝不靠近任意一端（防止混入真实的 ID 簇）
    lam = (torch.rand((num_gen, 1)) * 0.6 + 0.2).to(device)

    feats_a = []
    feats_b =[]

    for i in range(num_gen):
        c_a = class_a_list[i].item()
        c_b = class_b_list[i].item()
        idx_a = class_to_indices[c_a][torch.randint(len(class_to_indices[c_a]), (1,))].item()
        idx_b = class_to_indices[c_b][torch.randint(len(class_to_indices[c_b]), (1,))].item()
        feats_a.append(reps[idx_a])
        feats_b.append(reps[idx_b])

    feats_a = torch.stack(feats_a)
    feats_b = torch.stack(feats_b)

    # 1. 基础 Mixup 组合
    mixed_features = lam * feats_a + (1 - lam) * feats_b

    # 【妙招 2】：空间膨胀 (Noise Injection)
    # 计算 ID 特征的标准差，添加成比例的高斯噪声，把线段打散成“云团”
    reps_std = torch.std(reps, dim=0, keepdim=True)
    noise = torch.randn_like(mixed_features) * reps_std * 0.5 # 0.3是噪声强度，可以微调
    mixed_features = mixed_features + noise

    # # 【妙招 3】：特征范数对齐 (Norm Alignment)
    # # 如果提供了真实的 OOD 特征，我们就强行把合成特征的长度缩放到和真实 OOD 一样！
    # if real_ood_reps is not None and len(real_ood_reps) > 0:
    #     # 计算真实 OOD 的平均特征长度
    #     real_ood_norm = torch.norm(real_ood_reps, dim=1).mean()
    #     # 计算当前合成特征的长度
    #     syn_norm = torch.norm(mixed_features, dim=1, keepdim=True)
    #     # 将合成特征标准化后，缩放至真实 OOD 的长度
    #     mixed_features = (mixed_features / (syn_norm + 1e-8)) * real_ood_norm
    # 【优化妙招 3】：范数扰动对齐 (Norm Jittering Alignment)
    if real_ood_reps is not None and len(real_ood_reps) > 0:
        # 计算真实 OOD 的平均长度
        real_ood_norm = torch.norm(real_ood_reps, dim=1).mean()
        syn_norm = torch.norm(mixed_features, dim=1, keepdim=True)

        # --- 核心改动点 ---
        # 不要强行 1:1 对齐，给范数加一个 0.8 到 1.3 倍的随机抖动
        # 这样样本就会在径向（离原点的远近）上散开，形成一个厚实的带状区域，而不是一条细线
        jitter = (torch.rand((num_gen, 1), device=device) * 0.3 + 0.8)
        jitter = jitter*jitter
        target_norm = real_ood_norm * jitter

        mixed_features = (mixed_features / (syn_norm + 1e-8)) * target_norm
    else:
        # 如果没参考，直接随机推开
        mixed_features = mixed_features * (1 + torch.rand((num_gen, 1), device=device) * 0.3)
    # 4. 最后的扰动：让样本在目标区域“散开”，而不是一个死点
    reps_std = torch.std(reps, dim=0, keepdim=True)
    noise = torch.randn_like(mixed_features) * reps_std * 0.15 # 适度噪声
    mixed_features = mixed_features + noise

    return mixed_features


def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['model_name'], cfg['control_name']]
        if cfg['task']:
            model_tag_list += ['task']
            model_tag_list.append(str(cfg['task']))
        if cfg['seen_class_num']:
            model_tag_list += ['scn']
            model_tag_list.append(str(cfg['seen_class_num']))
        if cfg['adjust_classes']:
            model_tag_list += ['adjust_classes']
            model_tag_list.append(str(cfg['adjust_classes']))
        if cfg['ood_data']:
            model_tag_list += ['od']
            model_tag_list.append(str(cfg['ood_data']))
        if cfg['select_length']:
            model_tag_list += ['sl']
            model_tag_list.append(str(cfg['select_length']))
        if cfg['ood_thre']:
            model_tag_list += ['ot']
            model_tag_list.append(str(cfg['ood_thre']))
        if cfg['open_close_ratio']:
            model_tag_list += ['ocr']
            model_tag_list.append(str(cfg['open_close_ratio']))
        if cfg['sel_method']:
            model_tag_list += ['sm']
            model_tag_list.append(str(cfg['sel_method']))
        else:
            cfg['sel_method'] = 'None'
        if cfg['protection']:
            model_tag_list += ['pt']
            model_tag_list.append(str(cfg['protection']))
        model_tag_list +=['bm']
        if cfg['binary_method']:
            model_tag_list.append(str(cfg['binary_method']))
        if 'all_class_num' in cfg and cfg['all_class_num']:
            model_tag_list.append('acn')
            model_tag_list.append(str(cfg['all_class_num']))
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    dataset = fetch_dataset(cfg['data_name'])
    process_dataset(dataset)
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    metric = Metric({'test':['Loss', 'Accuracy']})
    result = resume(cfg['model_tag'], load_tag='checkpoint')
    last_epoch = result['epoch']
    supervised_idx = result['supervised_idx']
    data_split = result['data_split']
    model.load_state_dict(result['server'].model_state_dict)

    data_loader = make_data_loader(dataset, 'server')
    test_logger = make_logger(os.path.join('output', 'runs', 'test_{}'.format(cfg['model_tag'])))
    test_logger.safe(True)
    test_model = make_batchnorm_stats(dataset['train'], model, 'server')

    test(data_loader['test'], test_model, metric, test_logger, last_epoch)

    test_logger.safe(False)
    result = resume(cfg['model_tag'], load_tag='checkpoint')
    train_logger = result['logger'] if 'logger' in result else None
    result = {'cfg': cfg, 'epoch': last_epoch, 'supervised_idx': supervised_idx, 'data_split': data_split,
              'logger': {'train': train_logger, 'test': test_logger}}
    save(result, './output/result/{}.pt'.format(cfg['model_tag']))
    return


def test(data_loader, model, metric, logger, epoch):
    with torch.no_grad():
        model.train(False)
        all_entropies = []
        all_targets =[]
        all_reps = []
        all_probs =[]

        for i, input in enumerate(data_loader):
            input = collate(input)
            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
            all_targets.append(input['target'].cpu())
            input['target'].fill_(-1)
            output = model(input)

            all_reps.append(output['rep'].cpu())
            all_probs.append(output['target'].cpu())

            output['loss'] = torch.tensor(0.0, device=cfg['device'])
            evaluation = metric.evaluate(metric.metric_name['test'], input, output)
            logger.append(evaluation, 'test', input_size)

            logits = output['target']
            id_indices_list = [2, 3, 4, 5, 6, 7]
            if logits.shape[1] >= 8:
                 id_logits = logits[:, id_indices_list]
            else:
                 id_logits = logits

            probs = torch.softmax(id_logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)
            all_entropies.append(entropy.cpu())

        all_entropies = torch.cat(all_entropies)
        all_targets = torch.cat(all_targets)
        all_reps = torch.cat(all_reps)
        all_probs = torch.cat(all_probs)

        # -------------------------------------------------------------------------
        #[新增 T-SNE] 仅仅使用真实的全体数据进行拟合
        # -------------------------------------------------------------------------
        print(f"Generating T-SNE space strictly on real data for Epoch {epoch}...")

        X_real = all_reps.numpy()
        y_real = all_targets.numpy()

        tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42)
        X_embedded_real = tsne.fit_transform(X_real) # 这一步仅改变真实的降维流形

        # -------------------------------------------------------------------------
        # 从当前的 OOD 类别样本即时计算 Affinity Matrix
        # -------------------------------------------------------------------------
        id_indices = [0, 1, 2, 3, 4, 5]
        ood_indices = [6, 7, 8, 9]

        print(f"\nComputing Affinity Matrix from real OOD samples in current batch...")
        ood_mask_bool = np.isin(y_real, ood_indices)

        if ood_mask_bool.any():
            ood_logits = all_probs[ood_mask_bool][:, id_indices]
            ood_probs = torch.softmax(ood_logits, dim=1)
            affinity_matrix = torch.matmul(ood_probs.T, ood_probs) / ood_probs.size(0)
            affinity_matrix.fill_diagonal_(0)
            print("Successfully computed Affinity Matrix (OOD probability transposition mean).")
        else:
            print("Warning: No OOD samples found. Using uniform matrix.")
            affinity_matrix = torch.ones((len(id_indices), len(id_indices)))
            affinity_matrix.fill_diagonal_(0)

        # -------------------------------------------------------------------------
        # 将 affinity_matrix 映射填充到 10x10，防止类别 ID 直接索引越界
        # -------------------------------------------------------------------------
        full_aff = torch.zeros((10, 10), device='cpu')
        for i_idx, id_i in enumerate(id_indices):
            for j_idx, id_j in enumerate(id_indices):
                full_aff[id_i, id_j] = affinity_matrix[i_idx, j_idx].item()
        affinity_matrix_padded = full_aff

        # -------------------------------------------------------------------------
        # 提取真实 ID 数据生成合成特征
        # -------------------------------------------------------------------------
        print(f"Generating Synthetic OOD features using the Computed Matrix...")
        id_mask_bool_data = np.isin(y_real, id_indices)
        id_reps_real = all_reps[id_mask_bool_data]
        id_targets_real = all_targets[id_mask_bool_data]
        #[修改点]：把真实 OOD 也提出来，传进函数里作为“范数参考”
        real_ood_for_norm = all_reps[ood_mask_bool] if ood_mask_bool.any() else None

        synthetic_ood_reps = generate_batch_based_ood(
            reps=id_reps_real,
            targets=id_targets_real,
            affinity_matrix=affinity_matrix_padded,
            device='cpu',
            real_ood_reps=real_ood_for_norm,  # <--- 加上这个参数
            # fill_diagonal=False
        )
        X_syn = synthetic_ood_reps.numpy()

        # synthetic_ood_reps = generate_batch_based_ood(
        #     reps=id_reps_real,
        #     targets=id_targets_real,
        #     affinity_matrix=affinity_matrix_padded,
        #     device='cpu',
        #     mix_ratio=1.0,
        #     fill_diagonal=False
        # )
        X_syn = synthetic_ood_reps.numpy()

        # -------------------------------------------------------------------------
        # [核心神来之笔] 使用 KNN 模型，将高维合成特征映射到现有的 2D T-SNE 空间上
        # -------------------------------------------------------------------------
        print("Mapping synthetic points onto the existing T-SNE map using KNN...")
        knn = KNeighborsRegressor(n_neighbors=5)
        knn.fit(X_real, X_embedded_real)
        X_embedded_syn = knn.predict(X_syn)

        # -------------------------------------------------------------------------
        # 绘图部分：原始的点完全不受合成样本干扰
        # -------------------------------------------------------------------------
        plt.figure(figsize=(10, 8))
        cmap = plt.get_cmap('tab10')

        # 1. 画真实的 OOD (浅灰色圆点)
        if ood_mask_bool.any():
            plt.scatter(
                X_embedded_real[ood_mask_bool, 0],
                X_embedded_real[ood_mask_bool, 1],
                c='lightgray',
                label='Real OOD',
                alpha=0.6,
                s=20,
                edgecolors='grey',
                linewidths=0.5
            )

        # 2. 画真实的 ID (不同颜色的圆点)
        for i, class_idx in enumerate(id_indices):
            mask = (y_real == class_idx)
            if mask.sum() > 0:
                plt.scatter(
                    X_embedded_real[mask, 0],
                    X_embedded_real[mask, 1],
                    color=cmap(i % 10),
                    label=f'Class {class_idx}',
                    alpha=0.8,
                    s=20
                )

        # 3. [新增] 叠加上合成的 OOD 数据（暗灰色的星星）
        plt.scatter(
            X_embedded_syn[:, 0],
            X_embedded_syn[:, 1],
            c='dimgray',          # 深灰色
            marker='*',           # 星星形状
            label='Virtual OOD',
            alpha=0.8,
            s=60,                 # 设置稍大
            edgecolors='black',
            linewidths=0.5
        )

        # 隐藏坐标轴
        plt.xticks([])
        plt.yticks([])

        # 图例位置
        plt.legend(loc='upper right', framealpha=0.9)
        plt.tight_layout()
        save_path = f'tsne_epoch_{epoch}.png'
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"T-SNE saved to {save_path}")
        # -------------------------------------------------------------------------

        class_avg_entropies =[]
        print(f"\n======== Epoch {epoch} Entropy Analysis ========")
        for c in range(10):
            mask = (all_targets == c)
            if mask.sum() > 0:
                avg_ent = all_entropies[mask].mean().item()
            else:
                avg_ent = 0.0
            class_avg_entropies.append(avg_ent)
            print(f"Class {c} Avg Entropy: {avg_ent:.4f}")

        id_avg_entropy = sum(class_avg_entropies[i] for i in id_indices) / len(id_indices)
        ood_avg_entropy = sum(class_avg_entropies[i] for i in ood_indices) / len(ood_indices)

        print("-" * 40)
        print(f"ID  Classes {id_indices} Mean Entropy: {id_avg_entropy:.4f}")
        print(f"OOD Classes {ood_indices} Mean Entropy: {ood_avg_entropy:.4f}")
        print("============================================\n")

        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
    return

if __name__ == "__main__":
    main()