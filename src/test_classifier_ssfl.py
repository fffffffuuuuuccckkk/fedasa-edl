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
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    dataset = fetch_dataset(cfg['data_name'])
    process_dataset(dataset)
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    metric = Metric({'test': ['Loss', 'Accuracy']})
    result = resume(cfg['model_tag'], load_tag='best')
    last_epoch = result['epoch']
    print(result.keys())
    supervised_idx = result['supervised_idx']
    data_split = result['data_split']
    model.load_state_dict(result['server'].model_state_dict)
    data_loader = make_data_loader(dataset, 'server')
    test_logger = make_logger(os.path.join('output', 'runs', 'test_{}'.format(cfg['model_tag'])))
    test_logger.safe(True)
    test_model = make_batchnorm_stats(dataset['train'], model, 'server')
    test(data_loader['test'], test_model, metric, test_logger, last_epoch)
    import sys
    sys.exit()
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
        all_targets = []
        for i, input in enumerate(data_loader):
            input = collate(input)
            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
            all_targets.append(input['target'].cpu())
            input['target'].fill_(-1)
            output = model(input)
            # output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
            output['loss'] = torch.tensor(0.0, device=cfg['device'])
            evaluation = metric.evaluate(metric.metric_name['test'], input, output)
            logger.append(evaluation, 'test', input_size)
            # [新增] 计算当前 batch 的 Entropy
            # 假设 output['logits'] 包含模型输出的 logits
            logits = output['target']
            probs = torch.softmax(logits, dim=1)
            # Entropy公式: -sum(p * log(p))
            entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)

            # [新增] 将熵和对应的标签移回 CPU 并保存
            all_entropies.append(entropy.cpu())

        # [新增] 循环结束后进行统计和打印
        all_entropies = torch.cat(all_entropies)
        all_targets = torch.cat(all_targets)
        print(len(all_entropies),len(all_targets))

        class_avg_entropies = []
        print(f"\n======== Epoch {epoch} Entropy Analysis ========")
        # 假设总共10类 (根据前6后4的描述)
        for c in range(10):
            mask = (all_targets == c)
            if mask.sum() > 0:
                # 计算该类别所有样本的平均熵
                avg_ent = all_entropies[mask].mean().item()
            else:
                avg_ent = 0.0
            class_avg_entropies.append(avg_ent)
            print(f"Class {c} Avg Entropy: {avg_ent:.4f}")

        # 计算 ID (前6类: 0-5) 和 OOD (后4类: 6-9) 的平均值
        # 注意：这里计算的是"类平均的平均"，以平衡类别数量差异
        id_avg_entropy = sum(class_avg_entropies[:6]) / 6
        ood_avg_entropy = sum(class_avg_entropies[6:]) / 4

        print("-" * 40)
        print(f"ID  Classes (0-5) Mean Entropy: {id_avg_entropy:.4f}")
        print(f"OOD Classes (6-9) Mean Entropy: {ood_avg_entropy:.4f}")
        print("============================================\n")

        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
    return


if __name__ == "__main__":
    main()
