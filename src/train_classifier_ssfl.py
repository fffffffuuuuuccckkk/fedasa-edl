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
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score, average_precision_score, auc
import gc, ctypes
import faulthandler
faulthandler.enable()
def get_predictions(output_logits):
    """
    Converts logits to predicted class indices.
    Assumes output_logits is a torch.Tensor of shape (batch_size, num_classes).
    """
    return torch.argmax(output_logits, dim=1)

def compute_closed_acc_top_k(outputs_logits, targets, k=6):
    """
    Computes Closed-set Accuracy on the first K classes.
    """
    truncated_logits = outputs_logits[:, :k]
    preds_k = torch.argmax(truncated_logits, dim=1)
    mask = (targets < k)

    if not mask.any():
        return 0.0

    filtered_targets = targets[mask]
    filtered_preds = preds_k[mask]

    correct = (filtered_preds == filtered_targets).sum().item()
    total = len(filtered_targets)

    return correct / total if total > 0 else 0.0

def compute_open_acc(outputs_logits, targets, k_known=6):
    """
    计算 Open-Set Accuracy (Open-Acc)。
    """
    all_preds = torch.argmax(outputs_logits, dim=1)
    mapped_targets = targets.clone()
    unknown_mask = (targets >= k_known)
    mapped_targets[unknown_mask] = k_known

    mapped_preds = all_preds.clone()
    unknown_pred_mask = (all_preds >= k_known)
    mapped_preds[unknown_pred_mask] = k_known

    correct_predictions = (mapped_preds == mapped_targets).sum().item()
    total_samples = len(mapped_targets)

    return correct_predictions / total_samples if total_samples > 0 else 0.0


def compute_balanced_open_acc(outputs_logits, targets, k_known=6):
    """
    计算 Balanced Open-Set Accuracy (Balanced Open-Acc)。
    """
    all_preds = torch.argmax(outputs_logits, dim=1)
    mapped_targets = targets.clone()
    unknown_mask = (targets >= k_known)
    mapped_targets[unknown_mask] = k_known

    mapped_preds = all_preds.clone()
    unknown_pred_mask = (all_preds >= k_known)
    mapped_preds[unknown_pred_mask] = k_known

    class_accuracies =[]
    for cls_idx in range(k_known + 1):
        cls_mask = (mapped_targets == cls_idx)
        cls_total = cls_mask.sum().item()

        if cls_total > 0:
            cls_correct = (mapped_preds[cls_mask] == cls_idx).sum().item()
            cls_acc = cls_correct / cls_total
            class_accuracies.append(cls_acc)
        else:
            pass

    if len(class_accuracies) == 0:
        return 0.0

    return sum(class_accuracies) / len(class_accuracies)

def compute_oscr(id_preds, id_targets, id_confs, ood_confs):
    """
    计算 OSCR (Open Set Classification Rate).
    id_preds: ID 样本的预测类别 (numpy array)
    id_targets: ID 样本的真实类别 (numpy array)
    id_confs: ID 样本的置信度 (越高越被认为是 ID) (numpy array)
    ood_confs: OOD 样本的置信度 (越高越被认为是 ID) (numpy array)
    """
    if len(id_confs) == 0 or len(ood_confs) == 0:
        return 0.0

    is_correct = (id_preds == id_targets)

    # 组合 items: (置信度, 是否是OOD, 是否预测正确)
    # type: 0 表示 ID, 1 表示 OOD
    id_items =[(conf, 0, corr) for conf, corr in zip(id_confs, is_correct)]
    ood_items =[(conf, 1, False) for conf in ood_confs]
    all_items = id_items + ood_items

    # 按置信度降序排序
    all_items.sort(key=lambda x: x[0], reverse=True)

    fprs = [0.0]
    ccrs = [0.0]

    total_id = len(id_confs)
    total_ood = len(ood_confs)

    current_ood_count = 0
    current_correct_id_count = 0

    n = len(all_items)
    i = 0
    while i < n:
        current_conf = all_items[i][0]
        # 处理相同置信度的样本
        while i < n and all_items[i][0] == current_conf:
            if all_items[i][1] == 1:
                current_ood_count += 1
            elif all_items[i][1] == 0 and all_items[i][2]:
                current_correct_id_count += 1
            i += 1

        fprs.append(current_ood_count / total_ood)
        ccrs.append(current_correct_id_count / total_id)

    try:
        return auc(fprs, ccrs)
    except ValueError:
        return 0.0


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
            model_tag_list +=['ot']
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
        if 'fisher_c' in cfg and not cfg['fisher_c']:
            model_tag_list.append('nofc')
        elif 'fisher_c' in cfg and cfg['fisher_c'] != 0.01:
            model_tag_list.append('fc')
            model_tag_list.append(str(cfg['fisher_c']))
        if 'use_kl' in cfg and cfg['use_kl']:
            model_tag_list.append('uk')
            if cfg['use_kl'] !=1 :
                model_tag_list.append(str(cfg['use_kl']))
        if 'nouse_kl' in cfg and cfg['nouse_kl']:
            model_tag_list.append('nuk')
        if 'nouse_fisher' in cfg and cfg['nouse_fisher']:
            model_tag_list.append('nuf')
        if 'adj_fish' in cfg and cfg['adj_fish']:
            model_tag_list.append('af')
        if 'use_3sigma' in cfg and not cfg['use_3sigma']:
            model_tag_list.append('nu3')
        if 'kl_w' in cfg and cfg['kl_w'] != 0.05:
            model_tag_list.append('klw')
            model_tag_list.append(str(cfg['kl_w']))
        if 'nedl_server' in cfg and not cfg['nedl_server']:
            model_tag_list.append('nns')
        if 'nedl_client' in cfg and not cfg['nedl_client']:
            model_tag_list.append('nnc')
        if 'pedl_client' in cfg and not cfg['pedl_client']:
            model_tag_list.append('npc')
        if 'weight_mode' in cfg and cfg['weight_mode'] != 'polygamma':
            model_tag_list.append('wm')
            model_tag_list.append(cfg['weight_mode'])
        print(cfg['use_3sigma'])

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
        metric = Metric({'train':['Loss', 'Accuracy', 'PAccuracy', 'MAccuracy', 'LabelRatio'],
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
            best_balanced_open_acc = result.get('best_balanced_open_acc', 0)
            best_aupr      = result.get('best_aupr', 0)
            best_oscr      = result.get('best_oscr', 0)
        else:
            server = make_server(model)
            client = make_client(model, data_split)
            logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
            best_open_acc, best_close_acc, best_auroc, best_balanced_open_acc, best_aupr, best_oscr = 0, 0, 0, 0, 0, 0
    else:
        last_epoch = 1
        server = make_server(model)
        client = make_client(model, data_split)
        logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
        best_open_acc, best_close_acc, best_auroc, best_balanced_open_acc, best_aupr, best_oscr = 0, 0, 0, 0, 0, 0


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
        # metric1 = {'closed_acc': 0, 'open_acc': 0, 'auroc': 0, 'balanced_open_acc': 0, 'aupr': 0, 'oscr': 0}
        # test(data_loader['test'], model, metric, logger, epoch, metric1=metric1)
        model.load_state_dict(server.model_state_dict)
        test_model = make_batchnorm_stats(batchnorm_dataset, model, 'global')
        if cfg['task'] == 'OSR':
            test_model.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=False))
            server.model_state_dict = save_model_state_dict(test_model.state_dict())

        # 初始化 metric1 字典，加入 balanced_open_acc, aupr, oscr
        metric1 = {'closed_acc': 0, 'open_acc': 0, 'auroc': 0, 'balanced_open_acc': 0, 'aupr': 0, 'oscr': 0}
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
        is_best_aupr          = metric1.get('aupr', 0) > best_aupr
        is_best_oscr          = metric1.get('oscr', 0) > best_oscr

        # 更新最佳值
        best_close_acc         = max(metric1.get('closed_acc', 0), best_close_acc)
        best_open_acc          = max(metric1.get('open_acc', 0), best_open_acc)
        best_auroc             = max(metric1.get('auroc', 0), best_auroc)
        best_balanced_open_acc = max(metric1.get('balanced_open_acc', 0), best_balanced_open_acc)
        best_aupr              = max(metric1.get('aupr', 0), best_aupr)
        best_oscr              = max(metric1.get('oscr', 0), best_oscr)

        # 写入 result 供后续 resume 使用
        result['best_closed_acc']        = best_close_acc
        result['best_open_acc']          = best_open_acc
        result['best_auroc']             = best_auroc
        result['best_balanced_open_acc'] = best_balanced_open_acc
        result['best_aupr']              = best_aupr
        result['best_oscr']              = best_oscr

        print(f"Best closed acc: {best_close_acc:.4f}, Best open acc: {best_open_acc:.4f}, Best balanced open acc: {best_balanced_open_acc:.4f}, Best AUROC: {best_auroc:.4f}, Best AUPR: {best_aupr:.4f}, Best OSCR: {best_oscr:.4f}")

        # 保存常规 checkpoint
        chkpt_path = './output/model/{}_checkpoint.pt'.format(cfg['model_tag'])
        save(result, chkpt_path)

        # 分别保存达到最佳的多个独立的 pt 文件
        if is_best_closed:
            shutil.copy(chkpt_path, './output/model/{}_best_closed_acc.pt'.format(cfg['model_tag']))
        if is_best_open:
            shutil.copy(chkpt_path, './output/model/{}_best_open_acc.pt'.format(cfg['model_tag']))
        if is_best_auroc:
            shutil.copy(chkpt_path, './output/model/{}_best_auroc.pt'.format(cfg['model_tag']))
        if is_best_balanced_open:
            shutil.copy(chkpt_path, './output/model/{}_best_balanced_open_acc.pt'.format(cfg['model_tag']))
        if is_best_aupr:
            shutil.copy(chkpt_path, './output/model/{}_best_aupr.pt'.format(cfg['model_tag']))
        if is_best_oscr:
            shutil.copy(chkpt_path, './output/model/{}_best_oscr.pt'.format(cfg['model_tag']))

        # 保留原有的总体最佳判定（如果需要的话）
        if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            shutil.copy(chkpt_path, './output/model/{}_best.pt'.format(cfg['model_tag']))

        logger.reset()
        if epoch % 10 == 0:
            gc.collect()
            # 强制 glibc 释放内存回系统（非管理员环境下最有效的手段）
            try:
                ctypes.CDLL('libc.so.6').malloc_trim(0)
            except:
                pass
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
    info = {'info':['Model: {}'.format(cfg['model_tag']),
                     'Train Epoch (S): {}({:.0f}%)'.format(epoch, 100.),
                     'Learning rate: {:.6f}'.format(lr),
                     'Epoch Finished Time: {}'.format(epoch_finished_time)]}
    logger.append(info, 'train', mean=False)
    print(logger.write('train', metric.metric_name['train']))
    logger.safe(False)
    return


def test(data_loader, model, metric, logger, epoch, k_closed_acc=cfg['seen_class_num'],metric1=None):
    # 设置logger为安全模式，避免多线程问题
    logger.safe(True)

    # 初始化列表用于收集测试数据
    all_targets =[]  # 存储真实标签
    all_outputs_logits = []  # 存储模型输出的logits
    if 'openmatch' in cfg['loss_mode'] or 'iomatch' in cfg['loss_mode']:
        ovas =[]
    if 'divcon' in cfg['loss_mode']:
        all_head_logits =[]
    if 'edl' in cfg['loss_mode']:
        all_edl = []


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
            if 'edl' in cfg['loss_mode']:
                all_edl.append(output['edl_output'])
            # 如果是分布式训练，对损失求平均
            output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
            # 使用metric评估模型性能
            evaluation = metric.evaluate(metric.metric_name['test'], input, output)
            # 将评估结果记录到logger
            logger.append(evaluation, 'test', input_size)

        # 将收集到的数据拼接成张量
        if all_targets and all_outputs_logits and 'mix' in cfg['loss_mode'] and 'edl' not in cfg['loss_mode']:
            from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, roc_auc_score
            all_targets_tensor = torch.cat(all_targets, dim=0)  # 拼接所有真实标签
            all_outputs_logits_tensor = torch.cat(all_outputs_logits, dim=0)  # 拼接所有输出logits

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

            # --- 计算 AUROC ---
            from sklearn.metrics import roc_auc_score, average_precision_score
            ood_labels = (all_targets_tensor >= cfg['seen_class_num']).long().numpy()
            id_probs = torch.softmax(all_outputs_logits_tensor[:, :cfg['seen_class_num']], dim=1)
            max_id_probs, _ = torch.max(id_probs, dim=1)
            ood_scores = -max_id_probs.numpy()
            try:
                auroc = roc_auc_score(ood_labels, ood_scores)
            except ValueError:
                auroc = 0.0
            print(f"Open-Set Detection AUROC (Based on MSP): {auroc:.4f}")

            all_probs = torch.softmax(all_outputs_logits_tensor, dim=1)
            ood_scores_c1 = all_probs[:, cfg['seen_class_num']].numpy()
            try:
                auroc_c1 = roc_auc_score(ood_labels, ood_scores_c1)
            except ValueError:
                auroc_c1 = 0.0
            print(f"AUROC (基于 C+1 类概率): {auroc_c1:.4f}")
            auroc = max(auroc, auroc_c1)

            # --- 新增：计算 AUPR ---
            try:
                aupr_msp = average_precision_score(ood_labels, ood_scores)
            except ValueError:
                aupr_msp = 0.0
            print(f"Open-Set Detection AUPR (Based on MSP): {aupr_msp:.4f}")

            try:
                aupr_c1_score = average_precision_score(ood_labels, ood_scores_c1)
            except ValueError:
                aupr_c1_score = 0.0
            print(f"AUPR (基于 C+1 类概率): {aupr_c1_score:.4f}")
            aupr = max(aupr_msp, aupr_c1_score)

            # --- 新增：计算 OSCR ---
            id_mask_bool = (all_targets_tensor < cfg['seen_class_num']).numpy()
            ood_mask_bool = (all_targets_tensor >= cfg['seen_class_num']).numpy()

            if id_mask_bool.sum() > 0 and ood_mask_bool.sum() > 0:
                id_preds_np = all_predictions.numpy()[id_mask_bool]
                id_targets_np = all_targets_tensor.numpy()[id_mask_bool]

                # ood_scores 越大表示越可能是 OOD，取反变成置信度(越大越像 ID)
                id_confs_msp = -ood_scores[id_mask_bool]
                ood_confs_msp = -ood_scores[ood_mask_bool]
                oscr_msp = compute_oscr(id_preds_np, id_targets_np, id_confs_msp, ood_confs_msp)

                id_confs_c1 = -ood_scores_c1[id_mask_bool]
                ood_confs_c1 = -ood_scores_c1[ood_mask_bool]
                oscr_c1 = compute_oscr(id_preds_np, id_targets_np, id_confs_c1, ood_confs_c1)

                print(f"OSCR (Based on MSP): {oscr_msp:.4f}")
                print(f"OSCR (基于 C+1 类概率): {oscr_c1:.4f}")
                oscr = max(oscr_msp, oscr_c1)
            else:
                oscr = 0.0
                print("OSCR: 0.0000 (缺少 ID 或 OOD 样本)")

            # 存储指标
            if metric1 is not None:
                metric1['closed_acc'] = closed_acc
                metric1['open_acc'] = open_acc
                metric1['balanced_open_acc'] = balanced_open_acc
                metric1['auroc'] = auroc
                metric1['aupr'] = aupr
                metric1['oscr'] = oscr

        # elif 'edl' in cfg['loss_mode']:
        #     from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, roc_auc_score
        #     all_targets_tensor = torch.cat(all_targets, dim=0)
        #     all_edl_output_tensor = torch.cat(all_edl, dim=0)
        #     # 新增：获取 Softmax 分类头的输出
        #     K = cfg['seen_class_num']
        #     all_logits_tensor = torch.cat(all_outputs_logits, dim=0)[:,:K]

        #     # 1. 闭集逻辑：直接从 Softmax 分类头获取预测结果
        #     # probs_softmax = torch.softmax(all_logits_tensor, dim=1) # 如果需要概率
        #     _, y_pred_softmax = torch.max(all_logits_tensor, dim=1)

        #     # 2. EDL 逻辑：仅用于计算不确定性 (Uncertainty)
        #     alpha = all_edl_output_tensor

        #     S = torch.sum(alpha, dim=1, keepdim=True)
        #     uncertainty = K / S.squeeze() # u = K / S

        #     # --- [指标 A] 闭集准确率 (基于 Softmax 分类头) ---
        #     seen_mask = all_targets_tensor < K
        #     if seen_mask.sum() > 0:
        #         y_true_seen = all_targets_tensor[seen_mask].cpu().numpy()
        #         # 使用 Softmax 头的预测
        #         y_pred_seen = y_pred_softmax[seen_mask].cpu().numpy()
        #         closed_acc = accuracy_score(y_true_seen, y_pred_seen)

        #         cm_closed = confusion_matrix(y_true_seen, y_pred_seen, labels=list(range(K)))
        #         print("\n--- [Softmax Head] Closed-Set Confusion Matrix (ID Only) ---")
        #         print(cm_closed)
        #     else:
        #         closed_acc = 0.0

        #     # --- [指标 B] 开集判定逻辑 (Top-N 策略：Softmax 预测 + EDL 不确定性) ---
        #     num_ood_real = (all_targets_tensor >= K).sum().item()

        #     # 初始化开集预测：默认全部采用 Softmax 头的分类结果
        #     y_pred_open = y_pred_softmax.clone()

        #     if num_ood_real > 0:
        #         # 根据 EDL 的不确定性从大到小排序
        #         sorted_u_values, indices = torch.sort(uncertainty, descending=True)

        #         # 选取不确定性最高的前 Top-N 个样本，强制判定为未知类 K
        #         # 无论 Softmax 头原本预测是什么，这里都会被 EDL 的高不确定性覆盖
        #         ood_indices = indices[:num_ood_real]
        #         y_pred_open[ood_indices] = K

        #         dynamic_threshold = sorted_u_values[num_ood_real-1].item()
        #         print(f"\n[Hybrid Mode] 策略: Softmax分类 + EDL不确定性Top-N (N={num_ood_real})")
        #         print(f"[Hybrid Mode] 判定阈值: {dynamic_threshold:.4f}")

        #     # --- 最终评估 ---
        #     y_true_full = all_targets_tensor.clone()
        #     y_true_full[y_true_full >= K] = K

        #     y_true_full_np = y_true_full.cpu().numpy()
        #     y_pred_open_np = y_pred_open.cpu().numpy()

        #     cm_full = confusion_matrix(y_true_full_np, y_pred_open_np)
        #     print("\n--- [Hybrid] Full Confusion Matrix (Softmax+EDL) ---")
        #     print(cm_full)

        #     open_acc = accuracy_score(y_true_full_np, y_pred_open_np)
        #     balanced_open_acc = balanced_accuracy_score(y_true_full_np, y_pred_open_np)

        #     # AUROC 依然基于 EDL 的不确定性分数
        #     ood_labels = (all_targets_tensor >= K).long().numpy()
        #     ood_scores_u = uncertainty.cpu().numpy()
        #     try:
        #         auroc = roc_auc_score(ood_labels, ood_scores_u)
        #     except:
        #         auroc = 0.0

        #     print(f"\n--- [Hybrid] Performance Metrics ---")
        #     print(f"Closed-Set Acc (Softmax): {closed_acc:.4f}")
        #     print(f"Open-Set Acc (Top-N): {open_acc:.4f}")
        #     print(f"OOD Detection AUROC (EDL): {auroc:.4f}")

        #     if metric1 is not None:
        #         metric1.update({
        #             'closed_acc': closed_acc,
        #             'open_acc': open_acc,
        #             'balanced_open_acc': balanced_open_acc,
        #             'auroc': auroc
        #         })
        elif 'edl' in cfg['loss_mode']:
            from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score
            all_targets_tensor = torch.cat(all_targets, dim=0)
            all_edl_output_tensor = torch.cat(all_edl, dim=0)
            if False:
                # 1. 计算 EDL 的总证据 S 和不确定性 u = K / S
                # 1. 区分 ID 和 OOD 掩码
                # 假设类别总数为 K，标签 >= K 的为 OOD
                num_classes = all_edl_output_tensor.shape[1]
                seen_mask = all_targets_tensor < num_classes
                unseen_mask = all_targets_tensor >= num_classes

                # 2. 打印 ID 类前 50 个样本的 EDL 输出 (Alpha 向量)
                print("\n" + "="*50)
                print(f"--- First 50 ID Samples EDL Output (Alpha) ---")
                id_samples = all_edl_output_tensor[seen_mask]
                indices = torch.randperm(len(id_samples))[:50]
                print(id_samples[indices])
                if len(id_samples) > 0:
                    # 取前 50 个，如果不足 50 个则取全部
                    # print(id_samples[:100])
                    print(id_samples[indices])
                else:
                    print("No ID samples found.")

                # 3. 打印 OOD 类前 50 个样本的 EDL 输出 (Alpha 向量)
                print("\n" + "-"*50)
                print(f"--- First 50 OOD Samples EDL Output (Alpha) ---")
                ood_samples = all_edl_output_tensor[unseen_mask]
                if len(ood_samples) > 0:
                    # 取前 50 个，如果不足 50 个则取全部
                    indices = torch.randperm(len(ood_samples))[:50]
                    print(ood_samples[indices])
                else:
                    print("No OOD samples found.")
                print("="*50 + "\n")
                print(all_edl[-1])
                sys.exit()

            # K 是已知类的数量
            K = cfg['seen_class_num']
            # 获取 Softmax 分类头的 Logits (仅取前 K 类)
            all_logits_tensor = torch.cat(all_outputs_logits, dim=0)[:, :K]

            # 1. 闭集预测 (基于 Softmax 支路)
            probs_softmax = torch.softmax(all_logits_tensor, dim=1)
            max_probs_softmax, y_pred_softmax = torch.max(probs_softmax, dim=1)

            # 2. EDL 基础参数提取
            alpha = all_edl_output_tensor
            evidence = alpha - 1
            S_full = torch.sum(alpha, dim=1)
            num_classes = alpha.size(1)

            # --- [指标 A] 闭集混淆矩阵打印 ---
            seen_mask = (all_targets_tensor < K)
            if seen_mask.sum() > 0:
                y_true_seen = all_targets_tensor[seen_mask].cpu().numpy()
                y_pred_seen = y_pred_softmax[seen_mask].cpu().numpy()
                closed_acc = accuracy_score(y_true_seen, y_pred_seen)
                cm_closed = confusion_matrix(y_true_seen, y_pred_seen, labels=list(range(K)))
                print("\n--- [Softmax Head] Closed-Set Confusion Matrix ---")
                print(cm_closed)
            else:
                closed_acc = 0.0

            # --- [指标 B] 计算 6 种 OOD 评分准则 (分数越大表示越可能是 OOD) ---
            ood_labels = (all_targets_tensor >= K).long().numpy()
            scores_map = {}

            # (1) Vacuity: u = K/S
            # (1) Standard Vacuity: u = K / S_all
            scores_map['Vacuity_All'] = (K / S_full).cpu().numpy()

            # (2) Top-N Vacuity (处理长尾分布)
            # 获取排序后的 alpha
            sorted_alpha, _ = torch.sort(alpha, dim=1, descending=True)

            # Top-2/3 已知类
            k_23 = max(1, int(K * 2/3))
            S_top23 = torch.sum(sorted_alpha[:, :k_23], dim=1)
            scores_map['Vac_Top23'] = (k_23 / S_top23).cpu().numpy()

            # Top-1/3 已知类
            k_13 = max(1, K // 3)
            S_top13 = torch.sum(sorted_alpha[:, :k_13], dim=1)
            scores_map['Vac_Top13'] = (k_13 / S_top13).cpu().numpy()

            # Top-1/2 已知类
            k_12 = max(1, K // 2)
            S_top12 = torch.sum(sorted_alpha[:, :k_12], dim=1)
            scores_map['Vac_Top12'] = (k_12 / S_top12).cpu().numpy()
            # (2) Softmax MSP: -max(prob)
            scores_map['MSP'] = -max_probs_softmax.cpu().numpy()
            # (3) Max Evidence: -max(evidence)
            max_evid, _ = torch.max(evidence, dim=1)
            scores_map['MaxEvidence'] = -max_evid.cpu().numpy()
            # (4) Entropy (Dirichlet)
            entropy = torch.sum(torch.lgamma(alpha) - (alpha - 1.0) * torch.digamma(alpha), dim=1) + \
                      torch.lgamma(torch.sum(alpha, dim=1)) + \
                      (torch.sum(alpha, dim=1) - num_classes) * torch.digamma(torch.sum(alpha, dim=1))
            scores_map['Entropy'] = entropy.cpu().numpy()
            # (5) Dissonance (证据冲突度)
            norm_evid = evidence / (evidence + 1.0)
            sum_evid = torch.sum(norm_evid, dim=1, keepdim=True)
            diss = torch.sum(norm_evid * (torch.abs(sum_evid - norm_evid) / (sum_evid + 1e-8)), dim=1)
            scores_map['Dissonance'] = diss.cpu().numpy()
            # 方法 B: 标准 e/S 归一化方式 (Subjective Logic Beliefs)
            # S_full 是总证据量 (sum(e) + K)
            # 这里使用 S_full 或 torch.sum(evidence, dim=1) 取决于你是否包含先验 K
            # 按照标准 SL 理论，使用 S_full = sum(evidence) + K
            S_full_val = torch.sum(evidence, dim=1, keepdim=True) + K
            b_i = evidence / (S_full_val + 1e-8)
            sum_bi = torch.sum(b_i, dim=1, keepdim=True)

            # 计算 Dissonance1
            # 公式逻辑保持一致，但输入改为标准 Belief
            diss1 = torch.sum(b_i * (torch.abs(sum_bi - b_i) / (sum_bi + 1e-8)), dim=1)
            scores_map['Dissonance1'] = diss1.cpu().numpy()
            # (6) Energy Score
            scores_map['Energy'] = -torch.logsumexp(evidence, dim=1).cpu().numpy()

            weighted_conf = torch.max(alpha.cpu() * probs_softmax, dim=1)[0]
            scores_map['Weighted_Conf'] = -weighted_conf.cpu().numpy()

            # --- [寻找最优策略] ---
            results = {}
            for name, sc in scores_map.items():
                try:
                    results[name] = {
                        'auroc': roc_auc_score(ood_labels, sc),
                        'aupr': average_precision_score(ood_labels, sc),
                        'score_array': sc
                    }
                except:
                    results[name] = {'auroc': 0.0, 'aupr': 0.0, 'score_array': sc}

            if 'weight_mode' in cfg and cfg['weight_mode'] != 'polygamma':
                best_strategy = 'Dissonance1'
            else:
                best_strategy = max(results, key=lambda k: results[k]['auroc'])
            final_auroc = results[best_strategy]['auroc']
            final_aupr = results[best_strategy]['aupr']
            best_ood_score = results[best_strategy]['score_array']

            # --- [指标 C] 计算 OSCR ---
            id_mask_bool = (all_targets_tensor < K).numpy()
            ood_mask_bool = (all_targets_tensor >= K).numpy()
            if id_mask_bool.sum() > 0 and ood_mask_bool.sum() > 0:
                oscr = compute_oscr(y_pred_softmax[id_mask_bool].cpu().numpy(),
                                    all_targets_tensor[id_mask_bool].cpu().numpy(),
                                    -best_ood_score[id_mask_bool],
                                    -best_ood_score[ood_mask_bool])
            else:
                oscr = 0.0

            # --- [指标 D] 开集 Acc 与 混淆矩阵 (基于效果最好的指标进行 Top-N 判定) ---
            num_ood_real = int(ood_mask_bool.sum())
            y_pred_open = y_pred_softmax.clone()
            if num_ood_real > 0:
                # 转换回 tensor 以便使用 topk
                best_score_tensor = torch.from_numpy(best_ood_score).to(all_logits_tensor.device)
                _, top_indices = torch.topk(best_score_tensor, k=num_ood_real)
                y_pred_open[top_indices] = K

            y_true_full = all_targets_tensor.clone()
            y_true_full[y_true_full >= K] = K
            y_true_full_np = y_true_full.cpu().numpy()
            y_pred_open_np = y_pred_open.cpu().numpy()

            # 打印开集全量混淆矩阵
            cm_full = confusion_matrix(y_true_full_np, y_pred_open_np)
            print(f"\n--- [Hybrid] Full Confusion Matrix (Strategy: {best_strategy}) ---")
            print(cm_full)

            open_acc = accuracy_score(y_true_full_np, y_pred_open_np)
            balanced_open_acc = balanced_accuracy_score(y_true_full_np, y_pred_open_np)

            # --- 最终打印汇总 ---
            print(f"\n--- [Hybrid EDL Evaluation Summary] ---")
            print(f"Closed-Set Acc: {closed_acc:.4f}")
            print(f"Best OOD Strategy: {best_strategy}")
            print(f"Final AUROC:    {final_auroc:.4f}")
            print(f"Final AUPR:     {final_aupr:.4f}")
            print(f"OSCR:           {oscr:.4f}")
            print(f"Open-Set Acc:   {open_acc:.4f}")

            # 打印详细对比
            comparison = ", ".join([f"{k}: {v['auroc']:.3f}" for k, v in results.items()])
            print(f"AUROC Comparison: {comparison}")

            # 存储到 metric1 字典
            if metric1 is not None:
                metric1.update({
                    'closed_acc': closed_acc,
                    'open_acc': open_acc,
                    'balanced_open_acc': balanced_open_acc,
                    'auroc': final_auroc,
                    'aupr': final_aupr,
                    'oscr': oscr
                })
        elif 'bdmatch' in cfg['loss_mode']:
            all_targets_tensor = torch.cat(all_targets, dim=0)
            all_outputs_logits_tensor = torch.cat(all_outputs_logits, dim=0)

            probs = torch.sigmoid(all_outputs_logits_tensor)
            conf, y_pred = torch.max(probs, dim=1)

            seen_class_num = cfg['seen_class_num']
            seen_mask = all_targets_tensor < seen_class_num
            from sklearn.metrics import accuracy_score, balanced_accuracy_score

            if seen_mask.sum() > 0:
                y_true_seen = all_targets_tensor[seen_mask].numpy()
                y_pred_seen = y_pred[seen_mask].numpy()
                closed_acc = accuracy_score(y_true_seen, y_pred_seen)
            else:
                closed_acc = 0.0

            print(f"\n--- [BDMatch] Closed-Set Accuracy ---")
            print(f"Closed-Acc: {closed_acc:.4f}")

            threshold = 0.5
            is_unknown = conf < threshold
            y_pred_open = y_pred.clone()
            y_pred_open[is_unknown] = seen_class_num

            y_true_open = all_targets_tensor.clone()
            y_true_open[y_true_open >= seen_class_num] = seen_class_num

            open_acc = accuracy_score(y_true_open.numpy(), y_pred_open.numpy())
            balanced_open_acc = balanced_accuracy_score(y_true_open.numpy(), y_pred_open.numpy())

            print(f"\n--- [BDMatch] Open-Set Accuracy ---")
            print(f"Open-Acc (Threshold={threshold}): {open_acc:.4f}")

            # --- AUROC ---
            from sklearn.metrics import roc_auc_score, average_precision_score
            ood_labels = (all_targets_tensor >= seen_class_num).long().numpy()
            ood_scores = -conf.cpu().numpy()
            try:
                auroc = roc_auc_score(ood_labels, ood_scores)
            except ValueError:
                auroc = 0.0
            print(f"Open-Set Detection AUROC (Based on Max Sigmoid): {auroc:.4f}")

            # --- AUPR & OSCR ---
            try:
                aupr = average_precision_score(ood_labels, ood_scores)
            except ValueError:
                aupr = 0.0

            id_mask_bool = (all_targets_tensor < seen_class_num).numpy()
            ood_mask_bool = (all_targets_tensor >= seen_class_num).numpy()
            if id_mask_bool.sum() > 0 and ood_mask_bool.sum() > 0:
                id_preds_np = y_pred.numpy()[id_mask_bool]
                id_targets_np = all_targets_tensor.numpy()[id_mask_bool]
                id_confs = conf.cpu().numpy()[id_mask_bool]
                ood_confs = conf.cpu().numpy()[ood_mask_bool]
                oscr = compute_oscr(id_preds_np, id_targets_np, id_confs, ood_confs)
            else:
                oscr = 0.0

            if metric1 is not None:
                metric1['closed_acc'] = closed_acc
                metric1['open_acc'] = open_acc
                metric1['balanced_open_acc'] = balanced_open_acc
                metric1['auroc'] = auroc
                metric1['aupr'] = aupr
                metric1['oscr'] = oscr


        elif 'openmatch' in cfg['loss_mode'] or 'iomatch' in cfg['loss_mode']:
            all_targets_tensor = torch.cat(all_targets, dim=0)
            all_outputs_logits_tensor = torch.cat(all_outputs_logits, dim=0)
            all_ovas_tensor = torch.cat(ovas, dim=0)

            probs = torch.sigmoid(all_outputs_logits_tensor)
            conf, y_pred = torch.max(probs, dim=1)

            logits_ova = all_ovas_tensor
            inlier_pred = all_outputs_logits_tensor.argmax(dim=1)

            outlier_score = logits_ova[
                torch.arange(inlier_pred.size(0), device=all_outputs_logits_tensor.device),
                0,
                inlier_pred
            ]
            ood_mask = outlier_score > 0.5

            seen_class_num = cfg['seen_class_num']
            seen_idxs = all_targets_tensor < seen_class_num

            if seen_idxs.sum() > 0:
                y_true_seen = all_targets_tensor[seen_idxs].cpu().numpy()
                y_pred_seen = inlier_pred[seen_idxs].cpu().numpy()
                from sklearn.metrics import accuracy_score
                closed_acc = accuracy_score(y_true_seen, y_pred_seen)
            else:
                closed_acc = 0.0

            print(f"\n--- Closed-Set Accuracy (Seen Classes Only) ---")
            print(f"Closed-Acc: {closed_acc:.4f}")

            y_pred_open = inlier_pred.clone()
            y_pred_open[ood_mask] = seen_class_num

            y_true_open = all_targets_tensor.clone()
            y_true_open[y_true_open >= seen_class_num] = seen_class_num

            from sklearn.metrics import accuracy_score, balanced_accuracy_score
            open_acc = accuracy_score(y_true_open.cpu().numpy(), y_pred_open.cpu().numpy())
            balanced_open_acc = balanced_accuracy_score(y_true_open.cpu().numpy(), y_pred_open.cpu().numpy())

            print(f"\n--- Open-Set Accuracy (Balanced Accuracy) ---")
            print(f"Open-Acc: {open_acc:.4f}")

            # --- AUROC ---
            from sklearn.metrics import roc_auc_score, average_precision_score
            ood_labels = (all_targets_tensor >= seen_class_num).long().numpy()
            ood_scores_sig = -conf.cpu().numpy()
            try:
                auroc_sig = roc_auc_score(ood_labels, ood_scores_sig)
            except ValueError:
                auroc_sig = 0.0
            print(f"Open-Set Detection AUROC (Based on Max Sigmoid): {auroc_sig:.4f}")

            ood_scores_ova = outlier_score.cpu().numpy()
            try:
                auroc_ova = roc_auc_score(ood_labels, ood_scores_ova)
            except ValueError:
                auroc_ova = 0.0
            print(f"Open-Set Detection AUROC (Based on OVA Outlier Score): {auroc_ova:.4f}")
            auroc = min(auroc_sig, auroc_ova) # 遵循原有逻辑

            # --- AUPR & OSCR ---
            try:
                aupr1 = average_precision_score(ood_labels, ood_scores_ova)
            except ValueError:
                aupr1 = 0.0
            try:
                aupr2 = average_precision_score(ood_labels, ood_scores_sig)
            except ValueError:
                aupr2 = 0.0
            aupr = max(aupr1, aupr2)

            id_mask_bool = (all_targets_tensor < seen_class_num).numpy()
            ood_mask_bool = (all_targets_tensor >= seen_class_num).numpy()
            if id_mask_bool.sum() > 0 and ood_mask_bool.sum() > 0:
                id_preds_np = inlier_pred.cpu().numpy()[id_mask_bool]
                id_targets_np = all_targets_tensor.cpu().numpy()[id_mask_bool]

                id_confs1 = -ood_scores_ova[id_mask_bool]
                ood_confs1 = -ood_scores_ova[ood_mask_bool]
                oscr1 = compute_oscr(id_preds_np, id_targets_np, id_confs1, ood_confs1)

                id_confs2 = conf.cpu().numpy()[id_mask_bool]
                ood_confs2 = conf.cpu().numpy()[ood_mask_bool]
                oscr2 = compute_oscr(id_preds_np, id_targets_np, id_confs2, ood_confs2)

                oscr = max(oscr1, oscr2)
            else:
                oscr = 0.0

            if metric1 is not None:
                metric1['closed_acc'] = closed_acc
                metric1['open_acc'] = open_acc
                metric1['balanced_open_acc'] = balanced_open_acc
                metric1['auroc'] = auroc
                metric1['aupr'] = aupr
                metric1['oscr'] = oscr


        elif 'divcon' in cfg['loss_mode']:
            all_targets_tensor = torch.cat(all_targets, dim=0)
            all_outputs_logits_tensor = torch.cat(all_outputs_logits, dim=0)
            all_heads_tensor = torch.cat(all_head_logits, dim=0)

            probs = torch.sigmoid(all_outputs_logits_tensor)
            conf, y_pred = torch.max(probs, dim=1)

            probs_heads = torch.softmax(all_heads_tensor, dim=-1)
            probs_mean = torch.mean(probs_heads, dim=1)

            eps = 1e-8
            entropy_mean = -torch.sum(probs_mean * torch.log(probs_mean + eps), dim=1)
            entropy_heads = -torch.sum(probs_heads * torch.log(probs_heads + eps), dim=-1)
            mean_entropy = torch.mean(entropy_heads, dim=1)

            disagreement = entropy_mean - mean_entropy
            K = cfg.get('num_div_heads', 4)
            max_possible_score = torch.log(torch.tensor(float(K))).item()
            norm_scores = disagreement / max_possible_score
            outlier_scores = norm_scores

            seen_class_num = cfg['seen_class_num']

            seen_mask = all_targets_tensor < seen_class_num
            if seen_mask.sum() > 0:
                pred_seen = all_outputs_logits_tensor[seen_mask].argmax(dim=1)
                target_seen = all_targets_tensor[seen_mask]
                closed_acc = (pred_seen == target_seen).float().mean().item()
            else:
                closed_acc = 0.0

            print(f"\n--- [DivCon] Closed-Set Accuracy ---")
            print(f"Closed-Acc: {closed_acc:.4f}")

            from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score, average_precision_score

            ood_labels = (all_targets_tensor >= seen_class_num).long().numpy()
            ood_scores_np = outlier_scores.numpy()

            try:
                auroc_dc = roc_auc_score(ood_labels, ood_scores_np)
                print(f"Open-Set Detection AUROC (DivCon): {auroc_dc:.4f}")
            except:
                auroc_dc = 0.0

            ood_scores_sig = -conf.cpu().numpy()
            try:
                auroc_sig = roc_auc_score(ood_labels, ood_scores_sig)
            except ValueError:
                auroc_sig = 0.0
            print(f"Open-Set Detection AUROC (Based on Max Sigmoid): {auroc_sig:.4f}")

            auroc = min(auroc_dc, auroc_sig)

            pred_y = all_outputs_logits_tensor.argmax(dim=1)
            threshold = 0.5
            is_ood_pred = outlier_scores > threshold

            pred_y_open = pred_y.clone()
            pred_y_open[is_ood_pred] = seen_class_num

            true_y_open = all_targets_tensor.clone()
            true_y_open[true_y_open >= seen_class_num] = seen_class_num

            open_acc = accuracy_score(true_y_open.numpy(), pred_y_open.numpy())
            balanced_open_acc = balanced_accuracy_score(true_y_open.cpu().numpy(), pred_y_open.cpu().numpy())

            print(f"\n--- [DivCon] Open-Set Accuracy (Balanced Acc) ---")
            print(f"Open-Acc: {open_acc:.4f}")

            # --- AUPR & OSCR ---
            try:
                aupr_dc = average_precision_score(ood_labels, ood_scores_np)
            except:
                aupr_dc = 0.0
            try:
                aupr_sig = average_precision_score(ood_labels, ood_scores_sig)
            except:
                aupr_sig = 0.0
            aupr = max(aupr_dc, aupr_sig)

            id_mask_bool = (all_targets_tensor < seen_class_num).numpy()
            ood_mask_bool = (all_targets_tensor >= seen_class_num).numpy()
            if id_mask_bool.sum() > 0 and ood_mask_bool.sum() > 0:
                id_preds_np = pred_y.cpu().numpy()[id_mask_bool]
                id_targets_np = all_targets_tensor.cpu().numpy()[id_mask_bool]

                id_confs_dc = -ood_scores_np[id_mask_bool]
                ood_confs_dc = -ood_scores_np[ood_mask_bool]
                oscr_dc = compute_oscr(id_preds_np, id_targets_np, id_confs_dc, ood_confs_dc)

                id_confs_sig = conf.cpu().numpy()[id_mask_bool]
                ood_confs_sig = conf.cpu().numpy()[ood_mask_bool]
                oscr_sig = compute_oscr(id_preds_np, id_targets_np, id_confs_sig, ood_confs_sig)

                oscr = max(oscr_dc, oscr_sig)
            else:
                oscr = 0.0

            if metric1 is not None:
                metric1['closed_acc'] = closed_acc
                metric1['open_acc'] = open_acc
                metric1['auroc'] = auroc
                metric1['balanced_open_acc'] = balanced_open_acc
                metric1['aupr'] = aupr
                metric1['oscr'] = oscr

        # 记录和打印原始日志信息
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        # 打印测试结果
        print(logger.write('test', metric.metric_name['test']))

    logger.safe(False)
    return


if __name__ == "__main__":
    main()