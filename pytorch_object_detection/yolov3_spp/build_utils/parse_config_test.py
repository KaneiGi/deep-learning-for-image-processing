import os
import numpy as np


def parse_model_cfg(path: str):
    if not path.endswith('.cfg') or not os.path.exists(path):
        raise FileNotFoundError('the cfg file not exists')

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')

    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.strip() for x in lines]

    module_defs = []
    for line in lines:
        if line.startswith('['):
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].strip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0

        else:
            key, val = line.split('=')
            key, val = key.strip(), val.strip()

            if key == 'anchors':
                val = val.replace(' ', '')
                module_defs[-1][key] = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))
            elif (key in ['from', 'layers', 'mask']) or (key == 'size' and ',' in val):
                module_defs[-1][key] = [int(x) for x in val.split(',')]
            else:
                if val.isnumeric():
                    module_defs[-1][key] = int(val) if (int(val) - float(val)) == 0 else float(val)
                else:
                    module_defs[-1][key] = val

    supported = ['type', 'batch_normalize', 'filters', 'size', 'stride', 'pad', 'activation', 'layers', 'groups',
                 'from', 'mask', 'anchors', 'classes', 'num', 'jitter', 'ignore_thresh', 'truth_thresh', 'random',
                 'stride_x', 'stride_y', 'weights_type', 'weights_normalization', 'scale_x_y', 'beta_nms', 'nms_kind',
                 'iou_loss', 'iou_normalizer', 'cls_normalizer', 'iou_thresh', 'probability']

    for x in module_defs[1:]:
        for k in x:
            if k not in supported:
                raise ValueError('unsupported fields :{} in cfg'.format(k))

    return module_defs


def parse_data_cfg(path):
    if not os.path.exists(path) and os.path.exists('data' + os.sep + path):
        path = 'data' + os.sep + path

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    options = dict()

    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, val = line.split('=')
        options[key.strip()] = val.strip()

    return options
