import argparse
import datetime
import models
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from config import cfg, process_args
from data import fetch_dataset, split_dataset, make_data_loader, separate_dataset, separate_dataset_su, \
    make_batchnorm_dataset_su, make_batchnorm_stats
from metrics import Metric
from modules import Server, Client
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate
from logger import make_logger
from sklearn.metrics import confusion_matrix
from modules import save_model_state_dict
from collections import deque, defaultdict
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score
def get_predictions(output_logits):
    """
    Converts logits to predicted class indices.
    Assumes output_logits is a torch.Tensor of shape (batch_size, num_classes).
    """
    return torch.argmax(output_logits, dim=1)

def compute_closed_acc_top_k(outputs_logits, targets, k=6):
    """
    Computes Closed-set Accuracy on the first K classes.

    Args:
        outputs_logits (torch.Tensor): Logits from the model (N, C).
        targets (torch.Tensor): True class labels (N).
        k (int): Number of top classes to consider for closed-set.

    Returns:
        float: The closed-set accuracy for the top K classes.
    """
    # 1. Truncate logits to the first K classes
    # We only care about the predictions if the true label is within the first K classes.
    truncated_logits = outputs_logits[:, :k]

    # 2. Get predictions from the truncated logits
    # The max value within the first K classes determines the prediction.
    preds_k = torch.argmax(truncated_logits, dim=1)

    # 3. Filter for samples where the true target is in the first K classes (0 to K-1)
    mask = (targets < k)

    # If no targets are in the top K, accuracy is 0 (or we avoid division by zero)
    if not mask.any():
        return 0.0

    # 4. Filter true targets and predictions using the mask
    filtered_targets = targets[mask]
    filtered_preds = preds_k[mask]

    # 5. Compute accuracy (correct predictions / total relevant samples)
    correct = (filtered_preds == filtered_targets).sum().item()
    total = len(filtered_targets)

    return correct / total if total > 0 else 0.0

def compute_open_acc(outputs_logits, targets, k_known=6):
    """
    计算 Open-Set Accuracy (Open-Acc)。
    将所有大于等于 k_known 的真实标签视为“未知类” (索引 k_known)。

    Args:
        outputs_logits (torch.Tensor): Logits from the model (N, C).
        targets (torch.Tensor): True class labels (N).
        k_known (int): 已知类别的数量 (0 到 k_known - 1)。

    Returns:
        float: Open-set Accuracy.
    """

    # 1. 获取模型的预测结果 (通常是 logit 最大的索引)
    # 对于 Open-Set，我们假设模型输出的 C 个维度中，前 k_known 个是已知类。
    # 如果模型是 N+1 维输出（N已知类 + 1未知类），则需要根据实际模型调整。
    # 在这里，我们先基于原始 logits 获取预测索引。
    all_preds = torch.argmax(outputs_logits, dim=1)

    # 2. 重新映射真实标签 (Targets)
    # 所有真实标签 >= k_known 的样本，将其视为“未知类”，用 k_known 索引表示。
    # 索引 k_known 对应于 logits 矩阵中的第 k_known 列（如果模型输出是 C 维）。
    mapped_targets = targets.clone()

    # 找到所有未知类（标签 >= k_known）
    unknown_mask = (targets >= k_known)

    # 将未知类的真实标签重新映射为 k_known
    mapped_targets[unknown_mask] = k_known

    # 3. 重新映射预测结果 (Predictions)
    # 根据 OSR 评估惯例，通常需要一个阈值或单独的未知类得分。
    # 但根据你的描述“将closedacc和最后一类都收集起来，在closedacc之后所有的类都视为最后一类”，
    # 我们假设模型对于未知类别的预测是通过某种方式映射到索引 k_known（或其他预设索引）。

    # 如果模型是 N_classes > k_known 的，我们仍将所有预测结果 >= k_known 的索引也视为 k_known
    mapped_preds = all_preds.clone()

    # 找到所有模型预测为未知类的索引（索引 >= k_known）
    unknown_pred_mask = (all_preds >= k_known)

    # 将预测为未知类的索引重新映射为 k_known
    mapped_preds[unknown_pred_mask] = k_known


    # 4. 计算准确率 (Open-Acc)
    # Open-Acc 是在包含已知类和未知类（即 k_known + 1 类）的集合上的准确率。
    correct_predictions = (mapped_preds == mapped_targets).sum().item()
    total_samples = len(mapped_targets)

    return correct_predictions / total_samples if total_samples > 0 else 0.0


def compute_balanced_open_acc(outputs_logits, targets, k_known=6):
    """
    计算 Balanced Open-Set Accuracy (Balanced Open-Acc)。
    这实际上是计算映射后的 k_known + 1 个类别的 Macro-Average Recall。

    逻辑：
    1. 将所有 >= k_known 的真实标签映射为 k_known (未知类)。
    2. 将所有 >= k_known 的预测结果映射为 k_known (未知类)。
    3. 分别计算 0 到 k_known 每个类别的准确率 (Recall)。
    4. 对所有存在的类别的准确率求平均。

    Args:
        outputs_logits (torch.Tensor): 模型输出 Logits (N, C_out).
        targets (torch.Tensor): 真实标签 (N).
        k_known (int): 已知类别的数量 (0 到 k_known - 1)。

    Returns:
        float: Balanced Open-set Accuracy.
    """

    # 1. 获取模型的预测结果
    all_preds = torch.argmax(outputs_logits, dim=1)

    # 2. 重新映射真实标签 (Targets)
    # 将所有真实标签 >= k_known 的样本视为“未知类”，映射为 k_known
    mapped_targets = targets.clone()
    unknown_mask = (targets >= k_known)
    mapped_targets[unknown_mask] = k_known

    # 3. 重新映射预测结果 (Predictions)
    # 将所有预测结果 >= k_known 的样本视为“未知类”，映射为 k_known
    mapped_preds = all_preds.clone()
    unknown_pred_mask = (all_preds >= k_known)
    mapped_preds[unknown_pred_mask] = k_known

    # 4. 计算 Balanced Accuracy (Macro-Average of Recall)
    # 我们关注的类别集合是 0 到 k_known (共 k_known + 1 个类)
    class_accuracies = []

    # 遍历所有可能的类别 (包括最后那个合并后的未知类 k_known)
    for cls_idx in range(k_known + 1):
        # 找出当前类别 cls_idx 的所有真实样本索引
        cls_mask = (mapped_targets == cls_idx)

        # 计算该类别的样本总数
        cls_total = cls_mask.sum().item()

        if cls_total > 0:
            # 计算该类别预测正确的数量
            # 即：真实标签是 cls_idx 且 预测标签也是 cls_idx
            cls_correct = (mapped_preds[cls_mask] == cls_idx).sum().item()

            # 计算该类别的 Recall (准确率)
            cls_acc = cls_correct / cls_total
            class_accuracies.append(cls_acc)
        else:
            # 如果某个类别在当前 batch 或数据集中不存在，通常在计算 Balanced Acc 时会忽略它
            # 或者根据具体指标定义视为 0。这里采用忽略策略以避免拉低平均值。
            pass

    # 5. 求平均
    if len(class_accuracies) == 0:
        return 0.0

    return sum(class_accuracies) / len(class_accuracies)


cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = ['new',str(seeds[i]), cfg['data_name'], cfg['model_name'], cfg['control_name']]
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
        if 'weight' in cfg and cfg['weight']:
            model_tag_list.append('w')
            model_tag_list.append(str(cfg['weight']))
        if 'ssl_ratio' in cfg and cfg['ssl_ratio']:
            model_tag_list.append('sslr')
            model_tag_list.append(str(cfg['weight']))
        if 'client_noise_ood' in cfg and cfg['client_noise_ood']:
            model_tag_list.append('cno')
            model_tag_list.append(str(cfg['client_noise_ood']))
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[1])
    torch.manual_seed(cfg['seed'])
    if 'subspaces' in cfg['sel_method']:
        cfg['sel_method'] = cfg['sel_method'].replace('subspaces', '').strip('_').strip()
    torch.cuda.manual_seed(cfg['seed'])
    server_dataset = fetch_dataset(cfg['data_name'])
    client_dataset = fetch_dataset(cfg['data_name'])
    process_dataset(server_dataset)
    if cfg.get('ssl_ratio', 0) > 0:
        server_dataset['train'], client_dataset['train'], supervised_idx,server_dataset['unlabeled'] = separate_dataset_su(server_dataset['train'],
                                                                                           client_dataset['train'])
    else:
        server_dataset['train'], client_dataset['train'], supervised_idx,_ = separate_dataset_su(server_dataset['train'],
                                                                                           client_dataset['train'])
    data_loader = make_data_loader(server_dataset, 'global')
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    optimizer = make_optimizer(model.parameters(), 'local')
    scheduler = make_scheduler(optimizer, 'global')
    if cfg['sbn'] == 1:
        batchnorm_dataset = make_batchnorm_dataset_su(server_dataset['train'], client_dataset['train'])
    elif cfg['sbn'] == 0:
        batchnorm_dataset = server_dataset['train']
    else:
        raise ValueError('Not valid sbn')
    data_split = split_dataset(client_dataset, cfg['num_clients'], cfg['data_split_mode'])
    if cfg['loss_mode'] != 'sup':
        metric = Metric({'train': ['Loss', 'Accuracy', 'PAccuracy', 'MAccuracy', 'LabelRatio'],
                         'test': ['Loss', 'Accuracy']})
    else:
        metric = Metric({'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']})
    if cfg['resume_mode'] == 1:
        result = resume(cfg['model_tag'])
        last_epoch = result['epoch']
        if last_epoch > 1:
            data_split = result['data_split']
            supervised_idx = result['supervised_idx']
            server = result['server']
            client = result['client']
            optimizer.load_state_dict(result['optimizer_state_dict'])
            scheduler.load_state_dict(result['scheduler_state_dict'])
            logger = result['logger']

            best_close_acc = result.get('best_closed_acc', 0)
            best_open_acc  = result.get('best_open_acc', 0)
            best_auroc     = result.get('best_auroc', 0)
            best_balanced_open_acc = result.get('best_balanced_open_acc', 0)  # 新增
        else:
            server = make_server(model)
            client = make_client(model, data_split)
            logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
            best_open_acc, best_close_acc, best_auroc, best_balanced_open_acc = 0, 0, 0, 0
    else:
        last_epoch = 1
        server = make_server(model)
        client = make_client(model, data_split)
        logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
        best_open_acc, best_close_acc, best_auroc, best_balanced_open_acc = 0, 0, 0, 0


    for epoch in range(last_epoch, cfg['global']['num_epochs'] + 2):
        cfg['global_epoch'] = epoch
        train_client(batchnorm_dataset, client_dataset['train'], server, client, optimizer, metric, logger, epoch)

        if 'ft' in cfg and cfg['ft'] == 0:
            if cfg.get('ssl_ratio', 0) > 0:
                train_server(server_dataset['train'], server, optimizer, metric, logger, epoch, server_dataset['unlabeled'])
            else:
                train_server(server_dataset['train'], server, optimizer, metric, logger, epoch)
            import sys
            sys.exit()
            logger.reset()
            server.update_parallel(client)
        else:
            logger.reset()
            server.update(client)
            if cfg.get('ssl_ratio', 0) > 0:
                train_server(server_dataset['train'], server, optimizer, metric, logger, epoch, server_dataset['unlabeled'])
            else:
                train_server(server_dataset['train'], server, optimizer, metric, logger, epoch)
        scheduler.step()
        model.load_state_dict(server.model_state_dict)
        test_model = make_batchnorm_stats(batchnorm_dataset, model, 'global')
        if cfg['task'] == 'OSR':
            test_model.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=False))
            server.model_state_dict = save_model_state_dict(test_model.state_dict())
        # metric1 = {'closed_acc':0,'open_acc':0}
        # # if cfg['task'] == 'OSR':
        # test(data_loader['test'], test_model, metric, logger, epoch,metric1=metric1)

        # result = {'cfg': cfg, 'epoch': epoch + 1, 'server': server, 'client': client,
        #           'optimizer_state_dict': optimizer.state_dict(),
        #           'scheduler_state_dict': scheduler.state_dict(),
        #           'supervised_idx': supervised_idx, 'data_split': data_split, 'logger': logger}


        # best_close_acc = max(metric1['closed_acc'],best_close_acc)
        # best_open_acc = max(metric1['open_acc'],best_open_acc)

        # if 'best_closed_acc' not in result:
        #     result['best_closed_acc'] = best_close_acc
        # if 'best_open_acc' not in result:
        #     result['best_open_acc'] = best_open_acc
        # print('Best closed acc: {}, Best open acc: {}'.format(result['best_closed_acc'],result['best_open_acc']))
        # save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        # if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
        #     metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
        #     shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
        #                 './output/model/{}_best.pt'.format(cfg['model_tag']))

        # 初始化 metric1 字典，加入 balanced_open_acc
        metric1 = {'closed_acc': 0, 'open_acc': 0, 'auroc': 0, 'balanced_open_acc': 0}
        if 'mix' in cfg['loss_mode'] and cfg['task'] == 'OSR':
            test(data_loader['test'], test_model, metric, logger, epoch, metric1=metric1)
        elif (epoch%10 == 0) or (epoch == cfg['global']['num_epochs'] + 1):
            test(data_loader['test'], test_model, metric, logger, epoch, metric1=metric1)

        result = {'cfg': cfg, 'epoch': epoch + 1, 'server': server, 'client': client,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'scheduler_state_dict': scheduler.state_dict(),
                  'supervised_idx': supervised_idx, 'data_split': data_split, 'logger': logger}

        # 判断是否达到了各自的历史最佳
        is_best_closed        = metric1.get('closed_acc', 0) > best_close_acc
        is_best_open          = metric1.get('open_acc', 0) > best_open_acc
        is_best_auroc         = metric1.get('auroc', 0) > best_auroc
        is_best_balanced_open = metric1.get('balanced_open_acc', 0) > best_balanced_open_acc

        # 更新最佳值
        best_close_acc         = max(metric1.get('closed_acc', 0), best_close_acc)
        best_open_acc          = max(metric1.get('open_acc', 0), best_open_acc)
        best_auroc             = max(metric1.get('auroc', 0), best_auroc)
        best_balanced_open_acc = max(metric1.get('balanced_open_acc', 0), best_balanced_open_acc)

        # 写入 result 供后续 resume 使用
        result['best_closed_acc']        = best_close_acc
        result['best_open_acc']          = best_open_acc
        result['best_auroc']             = best_auroc
        result['best_balanced_open_acc'] = best_balanced_open_acc

        print(f"Best closed acc: {best_close_acc:.4f}, Best open acc: {best_open_acc:.4f}, Best balanced open acc: {best_balanced_open_acc:.4f}, Best AUROC: {best_auroc:.4f}")

        # 保存常规 checkpoint
        chkpt_path = './output/model/{}_checkpoint.pt'.format(cfg['model_tag'])
        save(result, chkpt_path)

        # 分别保存达到最佳的四个独立的 pt 文件
        if is_best_closed:
            shutil.copy(chkpt_path, './output/model/{}_best_closed_acc.pt'.format(cfg['model_tag']))
        if is_best_open:
            shutil.copy(chkpt_path, './output/model/{}_best_open_acc.pt'.format(cfg['model_tag']))
        if is_best_auroc:
            shutil.copy(chkpt_path, './output/model/{}_best_auroc.pt'.format(cfg['model_tag']))
        if is_best_balanced_open:
            shutil.copy(chkpt_path, './output/model/{}_best_balanced_open_acc.pt'.format(cfg['model_tag']))

        # 保留原有的总体最佳判定（如果需要的话）
        if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            shutil.copy(chkpt_path, './output/model/{}_best.pt'.format(cfg['model_tag']))



        logger.reset()
    return


def make_server(model):
    server = Server(model)
    return server


def make_client(model, data_split):
    client_id = torch.arange(cfg['num_clients'])
    client = [None for _ in range(cfg['num_clients'])]
    for m in range(len(client)):
        client[m] = Client(client_id[m], model, {'train': data_split['train'][m], 'test': data_split['test'][m]})
    return client


def train_client(batchnorm_dataset, client_dataset, server, client, optimizer, metric, logger, epoch):
    logger.safe(True)
    num_active_clients = int(np.ceil(cfg['active_rate'] * cfg['num_clients']))
    client_id = torch.arange(cfg['num_clients'])[torch.randperm(cfg['num_clients'])[:num_active_clients]].tolist()
    for i in range(num_active_clients):
        client[client_id[i]].active = True
    server.distribute(client, batchnorm_dataset)
    num_active_clients = len(client_id)
    start_time = time.time()
    lr = optimizer.param_groups[0]['lr']
    for i in range(num_active_clients):
        m = client_id[i]
        dataset_m = separate_dataset(client_dataset, client[m].data_split['train'])
        if 'batch' not in cfg['loss_mode'] and 'frgd' not in cfg['loss_mode'] and 'fmatch' not in cfg['loss_mode']:
            dataset_m = client[m].make_dataset(dataset_m, metric, logger)
        if dataset_m is not None:
            client[m].active = True
            client[m].train(dataset_m, lr, metric, logger)
        else:
            client[m].active = False
        if i % int((num_active_clients * cfg['log_interval']) + 1) == 0:
            _time = (time.time() - start_time) / (i + 1)
            epoch_finished_time = datetime.timedelta(seconds=_time * (num_active_clients - i - 1))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['global']['num_epochs'] - epoch) * _time * num_active_clients))
            exp_progress = 100. * i / num_active_clients
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch (C): {}({:.0f}%)'.format(epoch, exp_progress),
                             'Learning rate: {:.6f}'.format(lr),
                             'ID: {}({}/{})'.format(client_id[i], i + 1, num_active_clients),
                             'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']))
    # 恢复logger的安全模式设置
    logger.safe(False)
    return


def train_server(dataset, server, optimizer, metric, logger, epoch,unl_dataset=None):
    logger.safe(True)
    start_time = time.time()
    lr = optimizer.param_groups[0]['lr']
    if cfg.get('ssl_ratio', 0) > 0:
        server.train(dataset, lr, metric, logger, unl_dataset)
    else:
        server.train(dataset, lr, metric, logger)
    _time = (time.time() - start_time)
    epoch_finished_time = datetime.timedelta(seconds=round((cfg['global']['num_epochs'] - epoch) * _time))
    info = {'info': ['Model: {}'.format(cfg['model_tag']),
                     'Train Epoch (S): {}({:.0f}%)'.format(epoch, 100.),
                     'Learning rate: {:.6f}'.format(lr),
                     'Epoch Finished Time: {}'.format(epoch_finished_time)]}
    logger.append(info, 'train', mean=False)
    print(logger.write('train', metric.metric_name['train']))
    logger.safe(False)
    return


# def test(data_loader, model, metric, logger, epoch):
#     logger.safe(True)
#     with torch.no_grad():
#         model.train(False)
#         out_logits = []
#         for i, input in enumerate(data_loader):
#             input = collate(input)
#             input_size = input['data'].size(0)
#             input = to_device(input, cfg['device'])
#             output = model(input)
#             output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
#             evaluation = metric.evaluate(metric.metric_name['test'], input, output)
#             logger.append(evaluation, 'test', input_size)
#         info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
#         logger.append(info, 'test', mean=False)
#         print(logger.write('test', metric.metric_name['test']))
#     logger.safe(False)
#     return

import torch
from sklearn.metrics import confusion_matrix
import numpy as np

# --- Helper Functions (You would place these outside the main test function) ---

def get_predictions(output_logits):
    """
    Converts logits to predicted class indices.
    Assumes output_logits is a torch.Tensor of shape (batch_size, num_classes).
    """
    return torch.argmax(output_logits, dim=1)

def compute_closed_acc_top_k(outputs_logits, targets, k=6):
    """
    Computes Closed-set Accuracy on the first K classes.

    Args:
        outputs_logits (torch.Tensor): Logits from the model (N, C).
        targets (torch.Tensor): True class labels (N).
        k (int): Number of top classes to consider for closed-set.

    Returns:
        float: The closed-set accuracy for the top K classes.
    """
    # 1. Truncate logits to the first K classes
    # We only care about the predictions if the true label is within the first K classes.
    truncated_logits = outputs_logits[:, :k]

    # 2. Get predictions from the truncated logits
    # The max value within the first K classes determines the prediction.
    preds_k = torch.argmax(truncated_logits, dim=1)

    # 3. Filter for samples where the true target is in the first K classes (0 to K-1)
    mask = (targets < k)

    # If no targets are in the top K, accuracy is 0 (or we avoid division by zero)
    if not mask.any():
        return 0.0

    # 4. Filter true targets and predictions using the mask
    filtered_targets = targets[mask]
    filtered_preds = preds_k[mask]

    # 5. Compute accuracy (correct predictions / total relevant samples)
    correct = (filtered_preds == filtered_targets).sum().item()
    total = len(filtered_targets)

    return correct / total if total > 0 else 0.0

# --- Modified Test Function ---

def test(data_loader, model, metric, logger, epoch, k_closed_acc=cfg['seen_class_num'],metric1=None):
    # 设置logger为安全模式，避免多线程问题
    logger.safe(True)

    # 初始化列表用于收集测试数据
    all_targets = []  # 存储真实标签
    all_outputs_logits = []  # 存储模型输出的logits
    if 'openmatch' in cfg['loss_mode'] or 'iomatch' in cfg['loss_mode']:
        ovas = []
    if 'divcon' in cfg['loss_mode']:
        all_head_logits = []


    with torch.no_grad():
        # 将模型设置为评估模式
        model.train(False)
        for i, input in enumerate(data_loader):
            # 数据预处理和批处理
            input = collate(input)
            input_size = input['data'].size(0)

            # 收集真实标签数据
            if 'target' in input:
                all_targets.append(input['target'].cpu())  # 将标签移到CPU并添加到列表

            # 将输入数据移动到指定设备
            input = to_device(input, cfg['device'])
            # 如果是开集识别任务，将目标标签填充为-1
            if cfg['task'] == 'OSR' or cfg['seen_class_num']:
                input['target'].fill_(-1)
            # 模型前向传播
            output = model(input)

            # 收集模型输出的logits
            if 'bdmatch' in cfg['loss_mode']:
                # BDMatch 返回的是一个字典，主分支 logits 在 'logits' 键中
                if isinstance(output, dict) and 'logits' in output:
                    all_outputs_logits.append(output['logits'].cpu())
                elif 'target' in output:
                    # 如果你的模型 Wrapper 已经把 logits 也就是 target 字段
                    all_outputs_logits.append(output['target'].cpu())
            else:
                if 'target' in output:
                    all_outputs_logits.append(output['target'].cpu())  # 将logits移到CPU并添加到列表

            # 处理损失输出，确保存在loss字段
            if 'loss' not in output:
                output['loss'] = torch.tensor(0.0, device=cfg['device'])
            if 'openmatch' in cfg['loss_mode'] or 'iomatch' in cfg['loss_mode']:
                ovas.append(output['ova'])
            if 'divcon' in cfg['loss_mode'] and 'div_head_logits' in output:
                all_head_logits.append(output['div_head_logits'].cpu())
            # 如果是分布式训练，对损失求平均
            output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
            # 使用metric评估模型性能
            evaluation = metric.evaluate(metric.metric_name['test'], input, output)
            # 将评估结果记录到logger
            logger.append(evaluation, 'test', input_size)

        # 将收集到的数据拼接成张量
        if all_targets and all_outputs_logits and 'mix' in cfg['loss_mode']:
            all_targets_tensor = torch.cat(all_targets, dim=0)  # 拼接所有真实标签
            all_outputs_logits_tensor = torch.cat(all_outputs_logits, dim=0)  # 拼接所有输出logits
            if False:

                # ... (您的代码继续)
                # -----------------------------------------------------------
                # ⭐️ 插入逻辑：计算 Softmax 概率并打印类别平均最大概率 (原逻辑)
                # -----------------------------------------------------------

                # 获取已知类别数量
                seen_class_num = cfg['seen_class_num']
                N_total = len(all_targets_tensor) # 总样本数

                # 提取已知类别的logits（前seen_class_num列）
                id_logits = all_outputs_logits_tensor[:, :seen_class_num]

                # 对已知类别logits进行softmax变换，得到概率分布
                id_probabilities = torch.softmax(id_logits, dim=1)

                # 计算每个样本在已知类别上的最大softmax概率
                max_probabilities_id = torch.max(id_probabilities, dim=1).values

                # 使用defaultdict存储每个类别的最大概率列表
                class_max_probs = defaultdict(list)

                # 存储OOD（Out-of-Distribution）样本的索引
                ood_indices = []

                # 遍历所有样本，按真实标签进行分类统计
                for i in range(N_total):
                    target_label = all_targets_tensor[i].item()  # 获取真实标签
                    max_prob = max_probabilities_id[i].item()    # 获取最大概率值

                    if target_label >= 0: # 排除可能的填充值（如-1）
                        class_max_probs[target_label].append(max_prob)  # 按类别存储概率

                        # 如果标签大于等于已知类别数，认为是OOD样本
                        if target_label >= seen_class_num:
                            ood_indices.append(i) # 收集OOD样本的索引


                # 打印类别平均最大softmax概率阈值统计
                print("\n--- 类别平均最大 Softmax 概率阈值 (基于 ID 预测) ---")

                # 打印已知类别的平均阈值
                print("\n  [Seen Classes (ID)]")
                seen_avg_thresholds = {}
                for class_id in range(seen_class_num):
                    probs = class_max_probs.get(class_id, [])
                    if probs:
                        avg_prob = np.mean(probs)
                        seen_avg_thresholds[class_id] = avg_prob
                        print(f"    类别 {class_id} (ID): 平均最大 Prob = {avg_prob:.4f} (样本数: {len(probs)})")

                # 打印OOD类别的平均阈值
                print("\n  [OOD Classes (Novel/Unknown)]")
                ood_classes = [c for c in class_max_probs.keys() if c >= seen_class_num]

                for class_id in sorted(ood_classes):
                    probs = class_max_probs[class_id]
                    avg_prob = np.mean(probs)
                    print(f"    类别 {class_id} (OOD): 平均最大 Prob = {avg_prob:.4f} (样本数: {len(probs)})")

                # -----------------------------------------------------------
                # ⭐️ 核心修正逻辑：计算每个 ID 类别在所有样本上的第 1000 小值 (秩统计量)
                # -----------------------------------------------------------

                # 计算每个ID类别的秩统计量（第1000小的概率值）
                K = 1000 # 秩数

                print(f"\n--- 所有样本对每个 ID 类别概率的 第 {K} 小值 (秩统计量) ---")

                # 检查样本数量是否足够计算秩统计量
                if N_total < K:
                    print(f"警告: 总样本数 ({N_total}) 小于要求的秩数 ({K})，无法计算第 {K} 小值。")
                else:
                    # 遍历所有已知类别
                    for class_id in range(seen_class_num):
                        # 提取所有样本对该类别的概率值
                        probs_for_class_c = id_probabilities[:, class_id]

                        # 对概率值进行升序排序
                        sorted_probs, _ = torch.sort(probs_for_class_c)

                        # 获取第K小的值（索引K-1）
                        kth_smallest_value = sorted_probs[K - 1].item()

                        print(f"    类别 {class_id} (ID): 第 {K} 小值 = {kth_smallest_value:.6f}")

                # -----------------------------------------------------------
                # ⭐️ 插入逻辑结束
                # -----------------------------------------------------------

                # -----------------------------------------------------------
                # ⭐️ 新增核心逻辑：OOD 样本对 ID 类别的平均概率矩阵
                # -----------------------------------------------------------

                # 计算OOD样本对ID类别的平均softmax概率矩阵
                print("\n--- OOD 样本对 ID 类别输出的平均 Softmax 概率矩阵 ---")

                if not ood_indices:
                    print("警告: 测试集中未找到 OOD 样本，跳过计算。")
                else:
                    # 提取OOD样本的真实标签和概率
                    ood_targets = all_targets_tensor[ood_indices].numpy()
                    ood_probs = id_probabilities[ood_indices, :] # 形状: (N_ood, seen_class_num)

                    # 按OOD类别分组
                    ood_classes = sorted([c for c in class_max_probs.keys() if c >= seen_class_num])

                    # 设置输出表格的表头
                    header = ["OOD \\ ID"] + [f"ID {c}" for c in range(seen_class_num)]
                    row_format = "{:<12}" * (seen_class_num + 1)

                    # 打印表头和分隔线
                    print(row_format.format(*header))
                    print("-" * (12 * (seen_class_num + 1)))

                    # 遍历每个OOD类别
                    for ood_class_id in ood_classes:
                        # 创建当前OOD类别的样本掩码
                        class_mask = (ood_targets == ood_class_id)

                        if class_mask.sum() == 0:
                            continue

                        # 提取该OOD类别样本对所有ID类的概率输出
                        class_ood_probs = ood_probs[torch.from_numpy(class_mask).to(ood_probs.device), :]

                        # 计算对各个ID类别的平均概率
                        avg_probs_across_id = torch.mean(class_ood_probs, dim=0).cpu().numpy()

                        # 格式化并打印结果行
                        row = [f"OOD {ood_class_id}"] + [f"{p:.4f}" for p in avg_probs_across_id]
                        print(row_format.format(*row))

                # -----------------------------------------------------------
                # ⭐️ 新增逻辑结束
                # -----------------------------------------------------------

                # ... (第 1000 秩统计量的代码 - 保持不变)





            # 获取最终预测结果
            all_predictions = get_predictions(all_outputs_logits_tensor)

            # 转换为numpy数组用于sklearn混淆矩阵计算
            y_true = all_targets_tensor.numpy()  # 真实标签
            y_pred = all_predictions.numpy()     # 预测标签

            # 计算并打印混淆矩阵
            cm = confusion_matrix(y_true, y_pred)
            print("\n--- Confusion Matrix ---")
            print(cm)
            num_pred_seen = np.sum(y_pred < cfg['seen_class_num'])
            num_pred_unseen = np.sum(y_pred >= cfg['seen_class_num'])
            print(f"Predicted as Seen (In-Distribution):   {num_pred_seen}")
            print(f"Predicted as Unseen (Out-of-Distribution): {num_pred_unseen}")

            # 计算并打印闭集准确率（前K个类别）
            closed_acc = compute_closed_acc_top_k(all_outputs_logits_tensor, all_targets_tensor, k=k_closed_acc)

            print(f"\n--- Closed-Set Accuracy (Top {k_closed_acc} Classes) ---")
            print(f"Closed-Acc @ {k_closed_acc}: {closed_acc:.4f}")

            # 计算并打印开集准确率（已知类别+未知类别）
            open_acc = compute_open_acc(all_outputs_logits_tensor, all_targets_tensor, k_known=k_closed_acc)
            balanced_open_acc = compute_balanced_open_acc(all_outputs_logits_tensor, all_targets_tensor, k_known=k_closed_acc)

            print(f"\n--- Open-Set Accuracy (K_Known={k_closed_acc} + Unknown) ---")
            print(f"Open-Acc: {open_acc:.4f}")
            # 如果提供了metric1字典，存储准确率指标
            # if metric1 is not None:
            #     metric1['closed_acc'] = closed_acc
            #     metric1['open_acc'] = open_acc
            #     metric1['balanced_open_acc'] = balanced_open_acc

            # --- 新增：计算 AUROC ---
            from sklearn.metrics import roc_auc_score
            # OOD label: 1 if >= seen_class_num else 0
            ood_labels = (all_targets_tensor >= cfg['seen_class_num']).long().numpy()
            # ID 样本的最大 softmax 概率，概率越小越像 OOD，因此我们取负值作为 OOD score
            id_probs = torch.softmax(all_outputs_logits_tensor[:, :cfg['seen_class_num']], dim=1)
            max_id_probs, _ = torch.max(id_probs, dim=1)
            ood_scores = -max_id_probs.numpy()
            try:
                auroc = roc_auc_score(ood_labels, ood_scores)
            except ValueError:
                auroc = 0.0 # 测试集中没有OOD样本时防止报错
            print(f"Open-Set Detection AUROC (Based on MSP): {auroc:.4f}")

            all_probs = torch.softmax(all_outputs_logits_tensor, dim=1)

            ood_scores_c1 = all_probs[:, cfg['seen_class_num']].numpy()
            # C+1 类的概率本身就代表 OOD 置信度，概率越大越是 OOD，所以不需要取负号！

            try:
                auroc_c1 = roc_auc_score(ood_labels, ood_scores_c1)
            except ValueError:
                auroc_c1 = 0.0 # 测试集中没有 OOD 样本时防止报错
            print(f"AUROC (基于 C+1 类概率): {auroc_c1:.4f}")
            auroc = max(auroc,auroc_c1)

            # 如果提供了metric1字典，存储准确率指标
            if metric1 is not None:
                metric1['closed_acc'] = closed_acc
                metric1['open_acc'] = open_acc
                metric1['balanced_open_acc'] = balanced_open_acc
                metric1['auroc'] = auroc  # 补充存储 auroc



        elif 'bdmatch' in cfg['loss_mode']:
            all_targets_tensor = torch.cat(all_targets, dim=0)
            all_outputs_logits_tensor = torch.cat(all_outputs_logits, dim=0)

            # 1. 激活函数：BDMatch 使用 Sigmoid 而不是 Softmax
            probs = torch.sigmoid(all_outputs_logits_tensor)

            # 2. 获取每个样本的最大概率和对应的预测类别
            conf, y_pred = torch.max(probs, dim=1)

            # 3. Closed-Set Accuracy (只计算 ID 样本)
            seen_class_num = cfg['seen_class_num']
            seen_mask = all_targets_tensor < seen_class_num
            from sklearn.metrics import accuracy_score, balanced_accuracy_score

            if seen_mask.sum() > 0:
                # 提取 ID 样本的真实标签和预测
                y_true_seen = all_targets_tensor[seen_mask].numpy()
                y_pred_seen = y_pred[seen_mask].numpy()

                closed_acc = accuracy_score(y_true_seen, y_pred_seen)
            else:
                closed_acc = 0.0

            print(f"\n--- [BDMatch] Closed-Set Accuracy ---")
            print(f"Closed-Acc: {closed_acc:.4f}")

            # 4. Open-Set Accuracy
            # 核心逻辑：如果最大概率 < 0.5，则拒绝该样本（预测为 Unknown）
            # 0.5 是二分类的标准阈值
            threshold = 0.5
            is_unknown = conf < threshold

            # 构造 Open-Set 预测向量
            y_pred_open = y_pred.clone()
            # 将置信度低的样本标记为 Unknown (seen_class_num)
            y_pred_open[is_unknown] = seen_class_num

            # 构造 Open-Set 真实标签 (将所有 OOD 标签统一)
            y_true_open = all_targets_tensor.clone()
            y_true_open[y_true_open >= seen_class_num] = seen_class_num

            # 计算准确率
            open_acc = accuracy_score(y_true_open.numpy(), y_pred_open.numpy())
            balanced_open_acc = balanced_accuracy_score(y_true_open.numpy(), y_pred_open.numpy())

            print(f"\n--- [BDMatch] Open-Set Accuracy ---")
            print(f"Open-Acc (Threshold={threshold}): {open_acc:.4f}")

            # # 记录到 metric1
            # if metric1 is not None:
            #     metric1['closed_acc'] = closed_acc
            #     metric1['open_acc'] = open_acc
            #     metric1['balanced_open_acc'] = balanced_open_acc

            # --- 新增：计算 AUROC ---
            from sklearn.metrics import roc_auc_score
            ood_labels = (all_targets_tensor >= seen_class_num).long().numpy()
            # conf 是 max sigmoid，越低越像 OOD，取反作为 OOD 分数
            ood_scores = -conf.cpu().numpy()
            try:
                auroc = roc_auc_score(ood_labels, ood_scores)
            except ValueError:
                auroc = 0.0
            print(f"Open-Set Detection AUROC (Based on Max Sigmoid): {auroc:.4f}")

            # 记录到 metric1
            if metric1 is not None:
                metric1['closed_acc'] = closed_acc
                metric1['open_acc'] = open_acc
                metric1['balanced_open_acc'] = balanced_open_acc
                metric1['auroc'] = auroc  # 补充存储 auroc




        elif 'openmatch' in cfg['loss_mode'] or 'iomatch' in cfg['loss_mode']:
            # 1. 拼接 Tensor
            all_targets_tensor = torch.cat(all_targets, dim=0)
            all_outputs_logits_tensor = torch.cat(all_outputs_logits, dim=0)
            all_ovas_tensor = torch.cat(ovas, dim=0)  # 拼接 OVA 输出



            # 1. 激活函数：BDMatch 使用 Sigmoid 而不是 Softmax
            probs = torch.sigmoid(all_outputs_logits_tensor)

            # 2. 获取每个样本的最大概率和对应的预测类别
            conf, y_pred = torch.max(probs, dim=1)

            # 2. 处理 OVA Logits (参考 reference 逻辑)
            # view变换: (N, 2*C) -> (N, 2, C)，然后对 dim=1 (二分类维度) 做 softmax
            # index 0: 属于"非"该类的概率 (Outlier Score)
            # index 1: 属于该类的概率 (Inlier Score)
            logits_ova = all_ovas_tensor

            # 获取闭集预测结果 (0 ~ seen_class_num-1)
            inlier_pred = all_outputs_logits_tensor.argmax(dim=1)

            # 3. 计算 OOD 分数
            # 取出每个样本在其预测类别上的 "非" 概率 (outlier score)
            # 形状: [N]
            outlier_score = logits_ova[
                torch.arange(inlier_pred.size(0), device=all_outputs_logits_tensor.device),
                0,
                inlier_pred
            ]

            # 生成 OOD 掩码 (阈值通常设为 0.5)
            ood_mask = outlier_score > 0.5

            # -------------------------------------------
            # 计算 Closed-Set Accuracy (只看已知类样本)
            # -------------------------------------------
            seen_class_num = cfg['seen_class_num']
            # 筛选出真实标签属于已知类的样本索引
            seen_idxs = all_targets_tensor < seen_class_num

            if seen_idxs.sum() > 0:
                # 转换到 numpy 计算 accuracy_score
                y_true_seen = all_targets_tensor[seen_idxs].cpu().numpy()
                y_pred_seen = inlier_pred[seen_idxs].cpu().numpy()
                from sklearn.metrics import accuracy_score
                closed_acc = accuracy_score(y_true_seen, y_pred_seen)
            else:
                closed_acc = 0.0

            print(f"\n--- Closed-Set Accuracy (Seen Classes Only) ---")
            print(f"Closed-Acc: {closed_acc:.4f}")

            # -------------------------------------------
            # 计算 Open-Set Accuracy (Balanced Accuracy)
            # -------------------------------------------
            # 准备预测标签：将被判定为 OOD 的样本预测值设为 seen_class_num (即"未知"类)
            y_pred_open = inlier_pred.clone()
            y_pred_open[ood_mask] = seen_class_num

            # 准备真实标签：将所有 OOD 样本的标签统一设为 seen_class_num
            # 注意：如果 dataset 里的 unknown 标签已经是 seen_class_num 则不需要这一步，
            # 但为了安全起见，将所有 >= seen_class_num 的标签都归为一类
            y_true_open = all_targets_tensor.clone()
            y_true_open[y_true_open >= seen_class_num] = seen_class_num


            # open_acc = balanced_accuracy_score(y_true_open.cpu().numpy(), y_pred_open.cpu().numpy())
            from sklearn.metrics import accuracy_score,balanced_accuracy_score
            open_acc = accuracy_score(y_true_open.cpu().numpy(), y_pred_open.cpu().numpy())
            balanced_open_acc = balanced_accuracy_score(y_true_open.cpu().numpy(), y_pred_open.cpu().numpy())

            print(f"\n--- Open-Set Accuracy (Balanced Accuracy) ---")
            print(f"Open-Acc: {open_acc:.4f}")

            # -------------------------------------------
            # 记录指标
            # -------------------------------------------
            # if metric1 is not None:
            #     metric1['closed_acc'] = closed_acc
            #     metric1['open_acc'] = open_acc
            #     metric1['balanced_open_acc'] = balanced_open_acc

            # --- 新增：计算 AUROC ---
            from sklearn.metrics import roc_auc_score
            ood_labels = (all_targets_tensor >= seen_class_num).long().numpy()
            # conf 是 max sigmoid，越低越像 OOD，取反作为 OOD 分数
            ood_scores = -conf.cpu().numpy()
            try:
                auroc = roc_auc_score(ood_labels, ood_scores)
            except ValueError:
                auroc = 0.0
            print(f"Open-Set Detection AUROC (Based on Max Sigmoid): {auroc:.4f}")

            # # 记录到 metric1
            # if metric1 is not None:
            #     metric1['closed_acc'] = closed_acc
            #     metric1['open_acc'] = open_acc
            #     metric1['balanced_open_acc'] = balanced_open_acc
            #     metric1['auroc'] = auroc  # 补充存储 auroc

            # -------------------------------------------
            # 计算 AUROC
            # -------------------------------------------
            from sklearn.metrics import roc_auc_score
            ood_labels = (all_targets_tensor >= seen_class_num).long().numpy()
            # outlier_score 直接就是判断为“非某类”的分数，越高越像 OOD
            ood_scores = outlier_score.cpu().numpy()
            try:
                auroc1 = roc_auc_score(ood_labels, ood_scores)
            except ValueError:
                auroc1 = 0.0
            print(f"Open-Set Detection AUROC (Based on OVA Outlier Score): {auroc1:.4f}")
            auroc=min(auroc,auroc1)

            # -------------------------------------------
            # 记录指标
            # -------------------------------------------
            if metric1 is not None:
                metric1['closed_acc'] = closed_acc
                metric1['open_acc'] = open_acc
                metric1['balanced_open_acc'] = balanced_open_acc
                metric1['auroc'] = auroc  # 补充存储 auroc




        elif 'divcon' in cfg['loss_mode']:
            # 1. 拼接数据
            all_targets_tensor = torch.cat(all_targets, dim=0)
            all_outputs_logits_tensor = torch.cat(all_outputs_logits, dim=0)

            # 形状: [N, K_heads, C_classes]
            all_heads_tensor = torch.cat(all_head_logits, dim=0)

            # 1. 激活函数：BDMatch 使用 Sigmoid 而不是 Softmax
            probs = torch.sigmoid(all_outputs_logits_tensor)

            # 2. 获取每个样本的最大概率和对应的预测类别
            conf, y_pred = torch.max(probs, dim=1)

            # 2. 计算分歧度 (Disagreement / Consensus)
            # DivCon 论文通常使用 Jensen-Shannon Divergence (JSD) 或 预测方差
            # 这里实现标准的 JSD 作为 Disagreement (值越大，越可能是未知类)

            # (1) 计算每个头的 Softmax 概率: P_k
            probs_heads = torch.softmax(all_heads_tensor, dim=-1) # [N, K, C]

            # (2) 计算平均概率分布 (Consensus Probability): P_avg
            probs_mean = torch.mean(probs_heads, dim=1) # [N, C]

            # (3) 计算熵 H(P_avg)
            # 为了数值稳定，加个 eps
            eps = 1e-8
            entropy_mean = -torch.sum(probs_mean * torch.log(probs_mean + eps), dim=1) # [N]

            # (4) 计算平均熵 mean(H(P_k))
            entropy_heads = -torch.sum(probs_heads * torch.log(probs_heads + eps), dim=-1) # [N, K]
            mean_entropy = torch.mean(entropy_heads, dim=1) # [N]

            # (5) JSD = H(mean) - mean(H)
            # JSD 越大，说明头之间分歧越大 -> OOD 样本
            disagreement = entropy_mean - mean_entropy # [N]
            K = cfg.get('num_div_heads', 4)
            max_possible_score = torch.log(torch.tensor(float(K))).item()
            norm_scores = disagreement / max_possible_score
            outlier_scores = norm_scores



            # 3. 计算指标
            seen_class_num = cfg['seen_class_num']

            # --- Closed-Set Accuracy (仅已知类) ---
            seen_mask = all_targets_tensor < seen_class_num
            if seen_mask.sum() > 0:
                pred_seen = all_outputs_logits_tensor[seen_mask].argmax(dim=1)
                target_seen = all_targets_tensor[seen_mask]
                closed_acc = (pred_seen == target_seen).float().mean().item()
            else:
                closed_acc = 0.0

            print(f"\n--- [DivCon] Closed-Set Accuracy ---")
            print(f"Closed-Acc: {closed_acc:.4f}")

            # --- Open-Set Accuracy (Balanced Accuracy / AUROC) ---
            # 这是一个常用的简单评估方式：
            # 如果 outlier_score > 阈值，则预测为 unseen (label = seen_class_num)
            # 否则预测为 argmax(logits)

            # 为了计算准确率，我们需要确定一个阈值。
            # 在实际应用中通常使用验证集确定，或者计算 AUROC。
            # 这里为了和你的代码风格一致（计算 acc），我们可以计算 AUROC
            # 或者使用 OOD 样本和 ID 样本分数的分布来计算最佳分割点 (或固定阈值)。

            from sklearn.metrics import roc_auc_score, balanced_accuracy_score

            # 构造二分类标签：1为OOD，0为ID
            ood_labels = (all_targets_tensor >= seen_class_num).long().numpy()
            ood_scores_np = outlier_scores.numpy()

            # 计算 AUROC (衡量区分 ID 和 OOD 的能力)
            try:
                auroc = roc_auc_score(ood_labels, ood_scores_np)
                print(f"Open-Set Detection AUROC: {auroc:.4f}")
            except:
                auroc = 0.0


            # --- 新增：计算 AUROC ---
            from sklearn.metrics import roc_auc_score
            # conf 是 max sigmoid，越低越像 OOD，取反作为 OOD 分数
            ood_scores = -conf.cpu().numpy()
            try:
                auroc1 = roc_auc_score(ood_labels, ood_scores)
            except ValueError:
                auroc1 = 0.0
            print(f"Open-Set Detection AUROC (Based on Max Sigmoid): {auroc1:.4f}")

            auroc = min(auroc,auroc1)



            # 计算 Open-Set Accuracy (Balanced Accuracy)
            # 策略：取 Top 95% 的 ID 样本作为阈值 (TPR=95%)，或者简单的 Otsu/Mean
            # 这里演示使用 Balanced Accuracy（需要硬阈值）
            # 我们可以简单地假设 outlier_scores 归一化后的阈值，或者取中位数
            # DivCon 论文中往往结合 logits 和 disagreement，这里简化为只用 disagreement 过滤

            # 预测逻辑：
            pred_y = all_outputs_logits_tensor.argmax(dim=1)

            # 确定阈值：这里取所有样本分数的平均值作为临时阈值，实际应更严谨
            # 或者使用你在训练中用的 tau_open (如果保存了的话)

            threshold = 0.5

            is_ood_pred = outlier_scores > threshold

            # 修正预测：如果被判定为 OOD，则将预测类别设为 seen_class_num
            pred_y_open = pred_y.clone()
            pred_y_open[is_ood_pred] = seen_class_num

            # 修正真实标签：将所有 OOD 标签设为 seen_class_num
            true_y_open = all_targets_tensor.clone()
            true_y_open[true_y_open >= seen_class_num] = seen_class_num

            # open_acc = balanced_accuracy_score(true_y_open.numpy(), pred_y_open.numpy())
            from sklearn.metrics import accuracy_score, balanced_accuracy_score
            open_acc = accuracy_score(true_y_open.numpy(), pred_y_open.numpy())
            balanced_open_acc = balanced_accuracy_score(true_y_open.cpu().numpy(), pred_y_open.cpu().numpy())

            print(f"\n--- [DivCon] Open-Set Accuracy (Balanced Acc) ---")
            print(f"Open-Acc: {open_acc:.4f}")

            if metric1 is not None:
                metric1['closed_acc'] = closed_acc
                metric1['open_acc'] = open_acc
                metric1['auroc'] = auroc
                metric1['balanced_open_acc'] = balanced_open_acc

        # 记录和打印原始日志信息
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        # 打印测试结果
        print(logger.write('test', metric.metric_name['test']))

    logger.safe(False)
    return



if __name__ == "__main__":
    main()
