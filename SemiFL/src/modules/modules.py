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
from utils import to_device, make_optimizer, collate, to_device
from metrics import Accuracy
from collections import deque
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

    def distribute(self, client, batchnorm_dataset=None):
        model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        model.load_state_dict(self.model_state_dict)
        if batchnorm_dataset is not None:
            model = make_batchnorm_stats(batchnorm_dataset, model, 'global')
        model_state_dict = save_model_state_dict(model.state_dict())
        for m in range(len(client)):
            if client[m].active:
                client[m].model_state_dict = copy.deepcopy(model_state_dict)
        return

    def update(self, client):
        if 'fmatch' not in cfg['loss_mode']:
            with torch.no_grad():
                valid_client = [client[i] for i in range(len(client)) if client[i].active]
                if len(valid_client) > 0:
                    model = eval('models.{}()'.format(cfg['model_name']))
                    model.load_state_dict(self.model_state_dict)
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
                        for m in range(len(valid_client)):
                            self.prototypes.append(valid_client[m].prototype)
                            self.ood_threshold_mean += valid_client[m].ood_threshold_mean
                        self.ood_threshold_mean/=len(valid_client)
                        print('ood_threshold_SERVER:',self.ood_threshold_mean)
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
                            self.prototypes.append(valid_client[m].prototype)
                            self.ood_threshold_mean = 0
                            self.ood_threshold_mean += valid_client[m].ood_threshold_mean
                        self.ood_threshold_mean/=len(valid_client)
                        print('ood_threshold_SERVER:',self.ood_threshold_mean)
        else:
            raise ValueError('Not valid loss mode')
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

    def train(self, dataset, lr, metric, logger):
        if 'fmatch' not in cfg['loss_mode']:
            data_loader = make_data_loader({'train': dataset}, 'server')['train']
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

                    if cfg['task'] == 'OSR' and len(self.prototypes) != 0:
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

                                # 3. 将选取的 tensor 转移到正确的设备
                                proto_a = proto_a.to(cfg['device'])
                                proto_b = proto_b.to(cfg['device'])

                                # 4. 计算混合后的单个 prototype [D] 并添加到列表中
                                mixed_prototype = (lam_sample * proto_a + (1 - lam_sample) * proto_b).detach()
                                mixed_prototypes_list.append(mixed_prototype.unsqueeze(0)) # 变为 [1, D]

                            # 5. 使用 torch.cat() 沿着 Batch 维度堆叠起来
                            input['prototypes'] = torch.cat(mixed_prototypes_list, dim=0)
                            input['ood_threshold'] = self.ood_threshold_mean

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
            if cfg['select_length']:
                self.selected_ood_maxlength = cfg['select_length']
            self.selected_ood_update_length = 1
            self.selected_ood_count = 0
            self.selected_ood_scores = deque(maxlen=self.selected_ood_maxlength)
            self.selected_ood_labels = deque(maxlen=self.selected_ood_maxlength)
            self.selected_ood_images = deque(maxlen=self.selected_ood_maxlength)
            self.ood_threshold_mean = 0

    def make_hard_pseudo_label(self, soft_pseudo_label):
        max_p, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
        mask = max_p.ge(cfg['threshold'])
        return hard_pseudo_label, mask

    def make_dataset(self, dataset, metric, logger):
        if 'sup' in cfg['loss_mode']:
            return dataset
        elif 'fix' in cfg['loss_mode']:
            with torch.no_grad():
                data_loader = make_data_loader({'train': dataset}, 'global', shuffle={'train': False})['train']
                model = eval('models.{}(track=True).to(cfg["device"])'.format(cfg['model_name']))
                model.load_state_dict(self.model_state_dict)
                model.train(False)
                output = []
                target = []

                if cfg['task'] == 'OSR':
                    input_list = []
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    input = to_device(input, cfg['device'])
                    target_i = copy.deepcopy(input['target'])
                    if cfg['seen_class_num']:
                        input['target'].fill_(-1)
                    if cfg['task'] == 'OSR':
                        input_list.append(input)
                    output_ = model(input)
                    output_i = output_['target']
                    # target_i = input['target']
                    output.append(output_i.cpu())
                    target.append(target_i.cpu())
                output_, input_ = {}, {}
                output_['target'] = torch.cat(output, dim=0)
                if cfg['task'] == 'OSR':
                    output_['target'] = output_['target'][:,:-1]
                input_['target'] = torch.cat(target, dim=0)




                output_['target'] = F.softmax(output_['target'], dim=-1)
                new_target, mask = self.make_hard_pseudo_label(output_['target'])
                output_['mask'] = mask
                evaluation = metric.evaluate(['PAccuracy', 'MAccuracy', 'LabelRatio'], input_, output_)
                logger.append(evaluation, 'train', n=len(input_['target']))

                if cfg['task'] == 'OSR':
                # 1. 提取前面类别维度（排除最后一维OOD类）
                    id_probs = output_['target']  # shape: (N, C)，C=training_target_size

                    # 2. 计算每个样本的最大置信度（前面类别维度中的最大值）
                    max_id_probs, _ = torch.max(id_probs, dim=-1)  # shape: (N,)

                    # 3. 按最大置信度升序排序（取最低置信度样本）
                    _, sorted_indices = torch.sort(max_id_probs)  # 默认升序（小→大）

                    # 4. 截取前N个低置信度样本（N=selected_ood_maxlength）
                    top_indices = sorted_indices[:self.selected_ood_maxlength]
                    # 遍历排序后的索引，提取样本并填充队列
                    for idx in top_indices:
                        # 提取单个样本（input_list是batch列表，通过列表推导式展平为样本列表）
                        sample = {k: v[idx % len(v)] for batch in input_list for k, v in batch.items()}

                        self.selected_ood_images.append(sample)
                        self.selected_ood_labels.append(cfg['seen_class_num'])  # OOD标签
                        self.selected_ood_scores.append(max_id_probs[idx].item())  # 存储ID类最大概率




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
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model.parameters(), 'local')
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
                    input['mix_target'] = torch.stack([input['target'], input['mix_target']], dim=-1)
                    input['loss_mode'] = cfg['loss_mode']
                    input = to_device(input, cfg['device'])
                    if cfg['task'] == 'OSR':
                        if len(self.selected_ood_images) == self.selected_ood_images.maxlen:
                            input['selected_ood_images'] = self.selected_ood_images
                            input['selected_ood_labels'] = self.selected_ood_labels
                            input['selected_ood_scores'] = self.selected_ood_scores



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
        if cfg['task'] == 'OSR' and self.selected_ood_images is not None:


            data_tensors = [item['data'].unsqueeze(0).to(cfg['device']) for item in self.selected_ood_images]
            # 步骤 2: 使用 torch.cat() 在第一维 (dim=0) 上将这些张量拼接起来
            # 结果张量的第一维将是所有单独张量第一维之和 (16 + 32 + 8 = 56)
            collected_data_tensor = torch.cat(data_tensors, dim=0)
            input1 = {'data':collected_data_tensor,'target':torch.tensor(-1, device=cfg['device'])}
            out=model(input1)
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
            print(self.ood_threshold_mean)





            self.selected_ood_scores = deque(maxlen=self.selected_ood_maxlength)
            self.selected_ood_labels = deque(maxlen=self.selected_ood_maxlength)
            self.selected_ood_images = deque(maxlen=self.selected_ood_maxlength)
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
