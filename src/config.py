import yaml
import sys
global cfg
if 'cfg' not in globals():
    cmd_line_str = ' '.join(sys.argv)
    print(cmd_line_str)
    if '--task None' in cmd_line_str:
        with open('config2.yml', 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    elif 'mix' in cmd_line_str:
        with open('config.yml', 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    else:
        with open('config1.yml', 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)


def process_args(args):
    for k in cfg:
        cfg[k] = args[k]
    if 'control_name' in args and args['control_name'] is not None:
        control_name_list = args['control_name'].split('_')
        control_keys_list = list(cfg['control'].keys())
        cfg['control'] = {control_keys_list[i]: control_name_list[i] for i in range(len(control_name_list))}
    if cfg['control'] is not None:
        cfg['control_name'] = '_'.join([str(cfg['control'][k]) for k in cfg['control']])

    cfg['baselines'] = ['divcon', 'iomatch', 'bdmatch']
    return
