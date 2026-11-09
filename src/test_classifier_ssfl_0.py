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
import sys

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)

# ================= [新增] 虚拟 OOD 样本生成函数 =================
def generate_batch_based_ood(reps, targets, affinity_matrix, device, mix_ratio=1.0, fill_diagonal=False):
    """
    基于当前 Batch 的特征和选定的 Affinity Matrix 生成 OOD。
    仅在当前 Batch 存在的类别之间进行 Mixup。
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

    lam = torch.distributions.Beta(2.0, 2.0).sample((num_gen, 1)).to(device)

    feats_a =[]
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
        model_tag_list += ['bm']
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
    metric = Metric({'test': ['Loss', 'Accuracy']})
    # result = resume(cfg['model_tag'], load_tag='best')
    result = resume(cfg['model_tag'], load_tag='checkpoint')
    last_epoch = result['epoch']
    supervised_idx = result['supervised_idx']
    data_split = result['data_split']
    model.load_state_dict(result['server'].model_state_dict)
    # ================= [修改] 获取 Affinity Matrix =================
    server_obj = result['server']
    affinity_matrix = None
    if hasattr(server_obj, 'affinitys') and len(server_obj.affinitys) > 0:
        # server上的affinitys是一个list，对多个OOD类的矩阵求平均得到总的ID类间混淆矩阵
        affinity_matrix = torch.mean(torch.stack(server_obj.affinitys), dim=0).to(cfg['device'])
    else:
        # 兜底：如果没找到，默认一个均匀矩阵
        print("Warning: server.affinitys not found. Using uniform matrix.")
        affinity_matrix = torch.ones((6, 6)).to(cfg['device'])
    # ===============================================================
    data_loader = make_data_loader(dataset, 'server')
    test_logger = make_logger(os.path.join('output', 'runs', 'test_{}'.format(cfg['model_tag'])))
    test_logger.safe(True)
    test_model = make_batchnorm_stats(dataset['train'], model, 'server')
    test(data_loader['test'], test_model, metric, test_logger, last_epoch, affinity_matrix)
    test_logger.safe(False)
    result = resume(cfg['model_tag'], load_tag='checkpoint')
    train_logger = result['logger'] if 'logger' in result else None
    result = {'cfg': cfg, 'epoch': last_epoch, 'supervised_idx': supervised_idx, 'data_split': data_split,
              'logger': {'train': train_logger, 'test': test_logger}}
    save(result, './output/result/{}.pt'.format(cfg['model_tag']))
    return


def test(data_loader, model, metric, logger, epoch,affinity_matrix=None):
    with torch.no_grad():
        model.train(False)
        all_entropies = []
        all_targets = []
        all_reps = []  # [新增 T-SNE] 用于收集特征表示
        all_probs = []

        for i, input in enumerate(data_loader):
            input = collate(input)
            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
            all_targets.append(input['target'].cpu())
            input['target'].fill_(-1)
            output = model(input)

            # [新增 T-SNE] 收集当前 batch 的表示向量
            all_reps.append(output['rep'].cpu())
            all_probs.append(output['target'].cpu())

            output['loss'] = torch.tensor(0.0, device=cfg['device'])
            evaluation = metric.evaluate(metric.metric_name['test'], input, output)
            logger.append(evaluation, 'test', input_size)

            # 计算 Entropy
            # ================= [核心修改] =================
            logits = output['target']
            id_indices_list = [2, 3, 4, 5, 6, 7]
            # 1. 先取 ID 类的 Logits (只取第 2-7 列)
            # 注意：这里假设 logits 是 [Batch, 10]，请确保维度对应
            if logits.shape[1] >= 8: # 简单保护以防维度不足
                 id_logits = logits[:, id_indices_list]
            else:
                 id_logits = logits

            # 2. 再对 ID 类分布取 Softmax
            probs = torch.softmax(id_logits, dim=1)

            # 3. 再求熵
            entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)
            # ================= [核心修改结束] =================
            all_entropies.append(entropy.cpu())

        # 数据合并
        all_entropies = torch.cat(all_entropies)
        all_targets = torch.cat(all_targets)
        all_reps = torch.cat(all_reps)
        all_probs = torch.cat(all_probs) # 形状: [N, 10]

        print(len(all_entropies), len(all_targets))

        def compute_affinity(sample_indices, name):
            # 1. 定义那 6 个 ID 类的原始索引
            id_indices_list = [0, 1, 2, 3, 4, 5]

            # 2. 筛选样本行 (根据传入的 ID 或 OOD 样本索引)
            mask = np.isin(all_targets.cpu().numpy(), sample_indices)
            if not mask.any():
                print(f"Warning: No samples found for {name}")
                return None

            # 3. 截取原始输出的 2-7 列 [样本数, 6]
            # 注意：这里的 all_logits 必须是 model(input) 直接出的 raw logits
            sub_logits = all_probs[mask][:, id_indices_list]

            # 4. 对截取后的 6 个维度做 Softmax
            probs = torch.softmax(sub_logits, dim=1)

            # 5. 计算原始点积矩阵 (不进行任何归一化)
            # [6, 样本数] @ [样本数, 6] -> [6, 6]
            affinity_matrix = torch.matmul(probs.T, probs)

            # 6. 将对角线置为 0
            affinity_matrix.fill_diagonal_(0)

            # 打印结果，格式会和你最开始提供给我的那段字符串一致
            print(f"\n{name} Probability Affinity Matrix (6x6) (Epoch {epoch}):")
            print(affinity_matrix)

            # 保存 .pt 文件
            torch.save(affinity_matrix, f'{name}_aff_6x6_epoch_{epoch}.pt')
            return affinity_matrix





        if False:
            # 定义 ID 和 OOD
            id_indices = [2, 3, 4, 5, 6, 7]
            ood_indices = [0, 1, 8, 9]
            id_indices = [0, 1, 2, 3, 4, 5]
            ood_indices = [6, 7, 8, 9]
            # 分别计算 ID 和 OOD 的矩阵
            print("-" * 30)
            print("Computing Total ID Matrix...")
            id_affinity = compute_affinity(id_indices, "ID_Total")

            # 2. 计算 OOD 总体矩阵 (所有 OOD 混在一起)
            print("-" * 30)
            print("Computing Total OOD Matrix...")
            ood_affinity_total = compute_affinity(ood_indices, "OOD_Total")

            # 3. [新增] 循环遍历每个 OOD 类别，单独计算矩阵
            print("-" * 30)
            print("Computing Per-Class OOD Matrices...")
            for ood_cls in ood_indices:
                # 注意：传入的 sample_indices 必须是 list，所以用 [ood_cls]
                # 这里的逻辑是：只筛选出真实标签为 ood_cls 的样本，
                # 查看它们在 ID 类 (0-5) 上的共现/混淆情况
                compute_affinity([ood_cls], f"OOD_Class_{ood_cls}")
            # import sys
            # sys.exit()


        else:
            VIS_CLASSES = [0, 1, 2, 3, 4, 5]
            DISTANCES = np.arange(0.5, 3.01, 0.5)
            N_SAMPLES = 10

            rep_np = all_reps.cpu().numpy()
            target_np = all_targets.cpu().numpy()
            device = cfg['device']

            # ----------------------------
            # 1. 每类中心
            # ----------------------------
            centers = {}
            for cls in VIS_CLASSES:
                mask = (target_np == cls)
                if mask.sum() == 0:
                    continue
                centers[cls] = rep_np[mask].mean(axis=0)

            # ----------------------------
            # 2. 采样 & 置信度
            # ----------------------------
            mean_conf = {cls: [] for cls in VIS_CLASSES}
            all_scatter_x = {cls: [] for cls in VIS_CLASSES}
            all_scatter_y = {cls: [] for cls in VIS_CLASSES}

            model.eval()
            with torch.no_grad():
                for cls in VIS_CLASSES:
                    center = centers[cls]

                    for d in DISTANCES:
                        conf_list = []

                        for _ in range(N_SAMPLES):
                            direction = np.random.randn(center.shape[0])
                            direction /= (np.linalg.norm(direction) + 1e-8)

                            z = center + d * direction
                            z = torch.tensor(z, dtype=torch.float32).unsqueeze(0).to(device)

                            logits = model.classifier(z)   # ⚠️ 如需改名字，只改这一行
                            probs = torch.softmax(logits, dim=1)

                            conf = probs[0, cls].item()
                            conf_list.append(conf)

                            # 散点数据
                            all_scatter_x[cls].append(d)
                            all_scatter_y[cls].append(conf)

                        mean_conf[cls].append(np.mean(conf_list))

            # ----------------------------
            # 3. 可视化
            # ----------------------------
            plt.figure(figsize=(6.2, 4.5))
            cmap = plt.get_cmap("tab10")

            for i, cls in enumerate(VIS_CLASSES):
                color = cmap(i)

                # 散点（半透明）
                plt.scatter(
                    all_scatter_x[cls],
                    all_scatter_y[cls],
                    color=color,
                    alpha=0.25,
                    s=20
                )

                # 均值曲线
                plt.plot(
                    DISTANCES,
                    mean_conf[cls],
                    color=color,
                    linewidth=2.2,
                    marker='o',
                    label=f'Class {cls}'
                )

            plt.xlabel("Distance from class center")
            plt.ylabel("Confidence (GT class)")
            plt.xlim(0.45, 3.05)
            plt.ylim(0.0, 1.02)
            plt.legend(ncol=2, frameon=False)
            plt.tight_layout()
            plt.savefig("confidence_decay_scatter_cls0_5.png", dpi=300)
            plt.close()

            print("[Saved] confidence_decay_scatter_cls0_5.png")

            # ============================================================
            # import sys
            # sys.exit()











        # -------------------------------------------------------------------------
        # [新增 T-SNE] 可视化部分
        # -------------------------------------------------------------------------
        print(f"Generating T-SNE for Epoch {epoch}...")

        X = all_reps.numpy()
        y = all_targets.numpy()

        tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42)
        X_embedded = tsne.fit_transform(X)

        plt.figure(figsize=(10, 8))
        cmap = plt.get_cmap('tab10')

        # 画 OOD (灰色)
        id_indices = [0 , 1, 2, 3, 4, 5]
        ood_indices = [6, 7, 8, 9]
        # -------------------------------------------------------------------------
        #[核心修改] 从当前的 OOD 类别样本即时计算 Affinity Matrix
        # -------------------------------------------------------------------------
        print(f"\nComputing Affinity Matrix from real OOD samples in current batch...")
        ood_mask_bool = np.isin(all_targets.numpy(), ood_indices)

        if ood_mask_bool.any():
            # 1. 筛选出所有 OOD 样本在 ID 类别上的预测 Logits
            ood_logits = all_probs[ood_mask_bool][:, id_indices]
            # 2. 对这前 ID 类进行 Softmax 得到分布 P, shape =[N_ood, len(id_indices)]
            ood_probs = torch.softmax(ood_logits, dim=1)
            # 3. 和自身转置相乘求和取平均：(P^T @ P) / N_ood
            affinity_matrix = torch.matmul(ood_probs.T, ood_probs) / ood_probs.size(0)
            # 对角线置零以鼓励不同类别间 Mixup
            print(affinity_matrix)
            affinity_matrix.fill_diagonal_(0)
            print("Successfully computed Affinity Matrix (OOD probability transposition mean).")
        else:
            print("Warning: No OOD samples found. Using uniform matrix.")
            affinity_matrix = torch.ones((len(id_indices), len(id_indices)))
            affinity_matrix.fill_diagonal_(0)
        print(affinity_matrix)
        # -------------------------------------------------------------------------
        #[防越界保护] 将 affinity_matrix 映射填充到 10x10，防止类别 ID 直接索引越界
        # -------------------------------------------------------------------------
        full_aff = torch.zeros((10, 10), device='cpu')
        for i_idx, id_i in enumerate(id_indices):
            for j_idx, id_j in enumerate(id_indices):
                full_aff[id_i, id_j] = affinity_matrix[i_idx, j_idx].item()
        affinity_matrix_padded = full_aff

        # -------------------------------------------------------------------------
        # 使用真实 ID 数据与上一步计算出的 Affinity Matrix 合成虚拟 OOD 样本
        # -------------------------------------------------------------------------
        ood_mask = np.isin(y, ood_indices)
        if ood_mask.any():
            plt.scatter(
                X_embedded[ood_mask, 0],
                X_embedded[ood_mask, 1],
                c='lightgray',
                label='OOD', # 简化标签
                alpha=0.6,
                s=20,
                edgecolors='grey',
                linewidths=0.5
            )

        # 画 ID (不同颜色)
        for i, class_idx in enumerate(id_indices):
            mask = (y == class_idx)
            if mask.sum() > 0:
                plt.scatter(
                    X_embedded[mask, 0],
                    X_embedded[mask, 1],
                    color=cmap(i % 10),
                    label=f'Class {class_idx}', # 简化标签
                    alpha=0.8,
                    s=20
                )

        # ====== [修改区域开始] ======
        # 1. 去除标题
        # plt.title(f'T-SNE Visualization (Epoch {epoch})')

        # 2. 去除刻度
        plt.xticks([])
        plt.yticks([])
        # 如果你想连黑色边框也去掉，取消下面这行的注释:
        # plt.axis('off')

        # 3. 图例放在图片右上角 (loc='upper right')
        # framealpha=1 设置图例背景不透明，防止遮挡的点透过来干扰文字
        plt.legend(loc='upper right', framealpha=0.9)
        # ====== [修改区域结束] ======

        plt.tight_layout()
        save_path = f'tsne_epoch_{epoch}.png'
        # plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"T-SNE saved to {save_path}")
        # -------------------------------------------------------------------------

        # -------------------------------------------------------------------------
        # [新增] 使用真实 ID 数据与 Affinity Matrix 合成虚拟 OOD 样本
        # -------------------------------------------------------------------------
        print(f"Generating Synthetic OOD features using Affinity Matrix...")

        # 1. 取出所有真实 ID 样本特征和标签
        id_mask = np.isin(all_targets.numpy(), id_indices)
        id_reps_real = all_reps[id_mask]
        id_targets_real = all_targets[id_mask]

        # 2. 调用函数生成合成 OOD (为避免爆显存，在 CPU 上计算即可)
        synthetic_ood_reps = generate_batch_based_ood(
            reps=id_reps_real,
            targets=id_targets_real,
            affinity_matrix=affinity_matrix.cpu(),
            device='cpu',
            mix_ratio=1.0,
            fill_diagonal=False
        )

        # 3. 将真实特征和合成 OOD 特征拼接到一起供 t-SNE 使用
        X_real = all_reps.numpy()
        X_syn = synthetic_ood_reps.numpy()
        X_combined = np.concatenate([X_real, X_syn], axis=0)

        # 为了区分，赋予合成 OOD 类别一个伪标签（如 -1）
        y_real = all_targets.numpy()
        y_syn = np.full(X_syn.shape[0], -1)
        y_combined = np.concatenate([y_real, y_syn], axis=0)

        # -------------------------------------------------------------------------
        # [修改] T-SNE 可视化部分 (加入合成样本)
        # -------------------------------------------------------------------------
        print(f"Generating T-SNE for Epoch {epoch} with Synthetic OODs... (This might take a while)")

        tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42)
        # 对包含真实和合成的全体数据做 T-SNE
        X_embedded = tsne.fit_transform(X_combined)

        plt.figure(figsize=(10, 8))
        cmap = plt.get_cmap('tab10')

        # 1. 画真实的 OOD (灰色)
        ood_mask = np.isin(y_combined, ood_indices)
        if ood_mask.any():
            plt.scatter(
                X_embedded[ood_mask, 0],
                X_embedded[ood_mask, 1],
                c='lightgray',
                label='Real OOD',
                alpha=0.6,
                s=20,
                edgecolors='grey',
                linewidths=0.5
            )

        # 2. 画真实的 ID (不同颜色)
        for i, class_idx in enumerate(id_indices):
            mask = (y_combined == class_idx)
            if mask.sum() > 0:
                plt.scatter(
                    X_embedded[mask, 0],
                    X_embedded[mask, 1],
                    color=cmap(i % 10),
                    label=f'Class {class_idx}',
                    alpha=0.8,
                    s=20
                )

        # 3. [新增] 画合成的 OOD (红色星星)
        syn_mask = (y_combined == -1)
        if syn_mask.any():
            plt.scatter(
                X_embedded[syn_mask, 0],
                X_embedded[syn_mask, 1],
                c='crimson',      # 使用醒目的红色
                marker='*',       # 星星形状
                label='Synthetic OOD',
                alpha=0.7,
                s=40,             # 稍微大一点
                edgecolors='black',
                linewidths=0.5
            )

        plt.xticks([])
        plt.yticks([])

        plt.legend(loc='upper right', framealpha=0.9)

        plt.tight_layout()
        save_path = f'tsne_with_synthetic_ood_epoch_{epoch}.png'
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"T-SNE saved to {save_path}")
        # -------------------------------------------------------------------------

        class_avg_entropies = []
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