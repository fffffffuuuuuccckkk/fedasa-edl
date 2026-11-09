import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import models
from config import cfg, process_args
from data import fetch_dataset, make_data_loader, make_batchnorm_stats
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, resume, collate, load
from logger import make_logger
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import copy

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
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return
def runExperiment():
    print(cfg['loss_mode'])
    cfg['loss_mode'] = 'fix@0.95-mix'
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    dataset = fetch_dataset(cfg['data_name'])
    process_dataset(dataset)
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    batchnorm_dataset = dataset['train']
    metric = Metric({'test': ['Loss', 'Accuracy']})
    # result = resume(cfg['model_tag'], load_tag='best')
    model_tag = 'new_0_CIFAR10_wresnet28x2_150_fix@0.95-mix_100_0.1_iid_5-5_0.5_1_1_task_OSR_scn_6_adjust_classes_1_od_aug_sl_50_ocr_2_sm_None_bm_detach'
    model_tag = 'new_0_CIFAR10_wresnet28x2_150_fix@0.95-mix_100_0.1_non-iid-d-0.3_5-5_0.5_1_1_task_OSR_scn_6_adjust_classes_1_od_aug_sl_50_ocr_2_sm_subspaces1sentropyaffinity_bm_detach'
    tags = ['best_auroc','best','checkpoint','best_balanced_open_acc','best_closed_acc','best_open_acc']
    
    for load_tag in tags:
        result = load('./output/model/{}_{}.pt'.format(model_tag, load_tag))
        last_epoch = result['epoch']
        data_split = result['data_split']
        model.load_state_dict(result['server'].model_state_dict)
        data_loader = make_data_loader(dataset, 'server')
        test_logger = make_logger(os.path.join('output', 'runs', 'test_{}'.format(cfg['model_tag'])))
        test_logger.safe(True)
        test_model = make_batchnorm_stats(batchnorm_dataset, model, 'global')
        best_overall_aupr = 0.0
        best_overall_oscr = 0.0
        current_aupr, current_oscr = test(data_loader['test'], test_model, metric, test_logger, last_epoch)
        test_logger.safe(False)
        result = resume(cfg['model_tag'], load_tag='checkpoint')
        train_logger = result['logger'] if 'logger' in result else None
        result = {'cfg': cfg, 'epoch': last_epoch, 'data_split': data_split,
                'logger': {'train': train_logger, 'test': test_logger}}
        # print(result)
        if current_aupr > best_overall_aupr:
            best_overall_aupr = current_aupr
        if current_oscr > best_overall_oscr:
            best_overall_oscr = current_oscr
    print(f"\n{'='*30}")
    print(f"ALL TAGS COMPLETED")
    print(f"Best Global AUPR: {best_overall_aupr:.4f}")
    print(f"Best Global OSCR: {best_overall_oscr:.4f}")
    print(f"{'='*30}\n")
    print(f'{best_overall_aupr:.4f},{best_overall_oscr:.4f}')
    # save(result, './output/result/{}.pt'.format(cfg['model_tag']))
    return


def test(data_loader, model, metric, logger, epoch):
    all_targets = []
    all_logits = []

    with torch.no_grad():
        model.train(False)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input_size = input['data'].size(0)

            if 'target' in input:
                all_targets.append(copy.deepcopy(input['target'].cpu()))
                # 模仿第二个脚本：OSR 任务中将 target 填充为 -1 以防泄露
                input['target'].fill_(-1)

            input = to_device(input, cfg['device'])
            output = model(input)

            # 兼容不同模型的输出格式 (参考脚本2的 bdmatch/logits 处理)
            if isinstance(output, dict):
                logits = output.get('logits') if output.get('logits') is not None else output.get('target')
                if logits is not None:
                    all_logits.append(logits.cpu())
            elif isinstance(output, torch.Tensor):
                all_logits.append(output.cpu())

        # -----------------------------------------------------------
        # 计算逻辑开始
        # -----------------------------------------------------------
        aupr, oscr = 0.0, 0.0
        
        if all_targets and all_logits and 'seen_class_num' in cfg:
            all_targets_tensor = torch.cat(all_targets, dim=0)
            all_logits_tensor = torch.cat(all_logits, dim=0)
            seen_class_num = cfg['seen_class_num']
            
            # 过滤无效标签
            valid_mask = all_targets_tensor >= 0
            targets = all_targets_tensor[valid_mask].numpy()
            logits = all_logits_tensor[valid_mask]
            
            # 1. 提取 ID 预测和置信度 (基于已知类的 Softmax)
            id_logits = logits[:, :seen_class_num]
            id_probs = torch.softmax(id_logits, dim=1)
            max_id_probs, preds = torch.max(id_probs, dim=1)
            
            # 策略 A: 基于 MSP (Maximum Softmax Probability)
            # 概率越小，越可能是 OOD。对应 OOD Score 为 -max_id_probs
            ood_scores_msp = -max_id_probs.numpy()

            # 策略 B: 如果模型有 C+1 类 (专门的 OOD 检测位)
            # 模仿脚本2的 max(auroc, auroc_c1) 逻辑
            has_c1_head = logits.shape[1] > seen_class_num
            if has_c1_head:
                all_probs = torch.softmax(logits, dim=1)
                # 第 C+1 类的概率，越大越是 OOD
                ood_scores_c1 = all_probs[:, seen_class_num].numpy()
            else:
                ood_scores_c1 = ood_scores_msp # 如果没有，则相等

            # 准备标签
            is_ood = (targets >= seen_class_num).astype(int)
            is_id = (targets < seen_class_num)

            if is_ood.sum() > 0 and is_id.sum() > 0:
                # ==========================================
                # AUPR 计算 (取两种策略中表现最好的)
                # ==========================================
                aupr_msp = average_precision_score(is_ood, ood_scores_msp)
                aupr_c1 = average_precision_score(is_ood, ood_scores_c1) if has_c1_head else 0
                aupr = max(aupr_msp, aupr_c1)
                
                # ==========================================
                # OSCR 计算
                # ==========================================
                # OSCR 定义需区分 ID 样本中预测正确的子集
                id_targets = targets[is_id]
                id_preds = preds.numpy()[is_id]
                correct_id_mask = (id_preds == id_targets)
                
                # 计算 OSCR 函数
                def calculate_oscr(scores):
                    # scores 是越大概率越是 OOD 的分数
                    # 但 OSCR 传统公式通常用 Confidence (越大概率越是 ID)
                    # 我们转回 Confidence: conf = -score
                    confs = -scores 
                    id_confs = confs[is_id]
                    correct_id_confs = id_confs[correct_id_mask]
                    ood_confs = confs[is_ood == 1]
                    
                    if len(correct_id_confs) > 0:
                        # 正类: 正确分类的 ID (1), 负类: OOD (0)
                        y_true = [1] * len(correct_id_confs) + [0] * len(ood_confs)
                        y_score = list(correct_id_confs) + list(ood_confs)
                        auc = roc_auc_score(y_true, y_score)
                        return auc * (len(correct_id_confs) / len(id_confs))
                    return 0.0

                oscr_msp = calculate_oscr(ood_scores_msp)
                oscr_c1 = calculate_oscr(ood_scores_c1) if has_c1_head else 0
                oscr = max(oscr_msp, oscr_c1)

                print(f"--- Results [Best Strategy] ---")
                print(f"AUPR: {aupr:.4f} (MSP:{aupr_msp:.4f}, C1:{aupr_c1:.4f})")
                print(f"OSCR: {oscr:.4f} (MSP:{oscr_msp:.4f}, C1:{oscr_c1:.4f})")
            else:
                print("--- Warning: Missing ID or OOD samples ---")

        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)

    return aupr, oscr
if __name__ == "__main__":
    main()
