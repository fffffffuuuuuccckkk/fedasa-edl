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
        if cfg['select_length']:
            model_tag_list += ['sel_len']
            model_tag_list.append(str(cfg['select_length']))
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    server_dataset = fetch_dataset(cfg['data_name'])
    client_dataset = fetch_dataset(cfg['data_name'])
    process_dataset(server_dataset)
    server_dataset['train'], client_dataset['train'], supervised_idx = separate_dataset_su(server_dataset['train'],
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
        else:
            server = make_server(model)
            client = make_client(model, data_split)
            logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    else:
        last_epoch = 1
        server = make_server(model)
        client = make_client(model, data_split)
        logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    for epoch in range(last_epoch, cfg['global']['num_epochs'] + 1):
        train_client(batchnorm_dataset, client_dataset['train'], server, client, optimizer, metric, logger, epoch)
        if 'ft' in cfg and cfg['ft'] == 0:
            train_server(server_dataset['train'], server, optimizer, metric, logger, epoch)
            logger.reset()
            server.update_parallel(client)
        else:
            logger.reset()
            server.update(client)
            train_server(server_dataset['train'], server, optimizer, metric, logger, epoch)
        scheduler.step()
        model.load_state_dict(server.model_state_dict)
        test_model = make_batchnorm_stats(batchnorm_dataset, model, 'global')
        test(data_loader['test'], test_model, metric, logger, epoch)
        result = {'cfg': cfg, 'epoch': epoch + 1, 'server': server, 'client': client,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'scheduler_state_dict': scheduler.state_dict(),
                  'supervised_idx': supervised_idx, 'data_split': data_split, 'logger': logger}
        save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
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
    logger.safe(False)
    return


def train_server(dataset, server, optimizer, metric, logger, epoch):
    logger.safe(True)
    start_time = time.time()
    lr = optimizer.param_groups[0]['lr']
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

def test(data_loader, model, metric, logger, epoch, k_closed_acc=6):
    logger.safe(True)

    # 1. Initialize lists to collect data
    all_targets = []
    all_outputs_logits = []

    with torch.no_grad():
        model.train(False)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input_size = input['data'].size(0)

            # --- Collection: Get true labels (targets) ---
            # Assuming the true labels are stored in input['target']
            # Move target to CPU before appending to list
            if 'target' in input:
                all_targets.append(input['target'].cpu())

            input = to_device(input, cfg['device'])
            if cfg['task'] == 'OSR' or cfg['seen_class_num']:
                input['target'].fill_(-1)
            output = model(input)

            # --- Collection: Get model logits/predictions ---
            # Assuming the logits are stored in output['target'] or similar
            # If the key is 'target', it overrides the input's target, which might be confusing.
            # I will assume the key for the raw logits is 'output_logits' based on the original snippet's `out_logits = []`
            # and the user's request to collect 'output['target']'. I'll use 'output['target']'
            if 'target' in output:
                 # Move logits to CPU before appending to list
                all_outputs_logits.append(output['target'].cpu())

            if 'loss' not in output:
                output['loss'] = torch.tensor(0.0, device=cfg['device'])
            output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
            evaluation = metric.evaluate(metric.metric_name['test'], input, output)
            logger.append(evaluation, 'test', input_size)

        # 2. Concatenate collected data
        if all_targets and all_outputs_logits:
            all_targets_tensor = torch.cat(all_targets, dim=0)
            all_outputs_logits_tensor = torch.cat(all_outputs_logits, dim=0)

            # Get final predictions
            all_predictions = get_predictions(all_outputs_logits_tensor)

            # Convert to numpy for sklearn confusion matrix
            y_true = all_targets_tensor.numpy()
            y_pred = all_predictions.numpy()

            # 3. Calculate and Print Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            print("\n--- Confusion Matrix ---")
            print(cm)

            # 4. Calculate and Print Closed-Acc for Top K classes
            closed_acc = compute_closed_acc_top_k(all_outputs_logits_tensor, all_targets_tensor, k=k_closed_acc)

            print(f"\n--- Closed-Set Accuracy (Top {k_closed_acc} Classes) ---")
            print(f"Closed-Acc @ {k_closed_acc}: {closed_acc:.4f}")

            # 5. ⭐️ 新增：Calculate and Print Open-Acc
            open_acc = compute_open_acc(all_outputs_logits_tensor, all_targets_tensor, k_known=k_closed_acc)

            print(f"\n--- Open-Set Accuracy (K_Known={k_closed_acc} + Unknown) ---")
            print(f"Open-Acc: {open_acc:.4f}")

        # Original logging and printing
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))

    logger.safe(False)
    return



if __name__ == "__main__":
    main()
