import torch

from build_utils.layers import *
from build_utils.parse_config import *

ONNX_EXPORT = False


def create_modules(modules_defs: list, img_size):
    img_size = [img_size] * 2 if isinstance(img_size, int) else img_size
    modules_defs.pop(0)
    output_filters = [3]
    module_list = nn.ModuleList()
    routs = []
    yolo_index = -1

    for i, mdef in enumerate(modules_defs):
        modules = nn.Sequential()
        if mdef['type'] == 'convolutional':
            bn = mdef['batch_normalize']
            filters = mdef['filters']
            k = mdef['size']
            stride = mdef['stride'] if 'stride' in mdef else (mdef['stride_y'], mdef['stride_x'])
            if isinstance(k, int):
                modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                       out_channels=filters,
                                                       kernel_size=k,
                                                       stride=stride,
                                                       padding=(k // 2) if mdef['pad'] else 0,
                                                       bias=not bn))
            else:
                raise TypeError('conv2d filter must be a conv2d type')

            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters))

            if mdef['activation'] == 'leaky':
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))

            else:
                pass

        elif mdef['type'] == 'maxpool':
            k = mdef['size']
            stride = mdef['stride']
            modules = nn.MaxPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)

        elif mdef['upsample']:
            if ONNX_EXPORT:
                pass
            else:
                modules == nn.Upsample(scale_factor=mdef['stride'])

        elif mdef['type'] == 'route':
            layers = mdef['layers']
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat(layers=layers)

        elif mdef['type'] == 'shortcut':
            layers = mdef['from']
            filters = output_filters[-1]
            routs.append(i + layers[0])
            modules = WeightedFeatureFusion(layers=layers, weight='weight_type' in mdef)

        elif mdef['type'] == 'yolo':
            yolo_index += 1
            stride = [32, 16, 8]

            modules = YOLOLayer(anchors=mdef['anchors'][mdef['mask']],
                                nc=mdef['classes'],
                                img_size=img_size,
                                stride=stride[yolo_index])

            try:
                j = -1
                bias_ = module_list[j][0].bias
                bias = bias_.view(modules.na, -1)
                bias[:, 4] += -4.5
                bias[:, 5:] += math.log(0.6 / (modules.nc - 0.99))
                module_list[j][0].bias = torch.nn.Parameter(bias_, requires_grad=bias_.requires_grad)
            except Exception as e:
                print('Warning: smart bias initialization fails.', e)

        else:
            print("Warning: Unrecognized Layer Type: " + mdef["type"])

        module_list.append(modules)
        output_filters.append(filters)

    routs_binary = [False] * len(modules_defs)
    for i in routs:
        routs_binary[i] = True
    return module_list, routs_binary


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, img_size, stride):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.stride = stride
        self.na = len(anchors)
        self.nc = nc
        self.no = self.nc + 5
        self.nx, self.ny, self, ng = 0, 0, (0, 0)
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)
        self.grid = None

    def create_grids(self, ng=(13, 13), device='cpu'):
        self.nx, self.ny = ng
        self.ng = torch.tensor(ng, dtype=torch.float)
        if not self.training:
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device),
                                     torch.arange(self.nx, device=device)])
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()
        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p):

        batch_size, _, ny, nx = p.shape
        if (self.nx, self.ny) != (nx, ny) or self.grid is None:
            self.create_grids((nx, ny), p.device)

        p = p.view(batch_size, self.na, self.no, self.nx, self.ny).perform(0, 1, 3, 4, 2).contiguous()

        if self.training:
            return p
        else:
            io = p.clone()
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh
            io[..., :4] *= self.stride
            torch.sigmoid_(io[..., 4:])
            return io.view(batch_size, -1, self.no), p


class Darknet(nn.Module):

    def __init__(self, cfg, img_size=(416, 416), verbose=False):
        super(Darknet, self).__init__()
        self.input_size = [img_size] * 2 if isinstance(img_size, int) else img_size
        self.modules_defs = parse_model_cfg(cfg)
        self.module_list, self.routs = create_modules(self.modules_defs, img_size)
        self.yolo_layers = get_yolo_layers(self)
        self.info(verbose)

    def forward(self, x, verbose=False):
        return self.forward_once(x, verbose=verbose)

    def forward_once(self, x, verbose=False):

        yolo_out, out = [], []
        if verbose:
            print('0', x.shape)
            str = ''

        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            if name in ['WeightedFeatureFusion', 'FeatureConcat']:
                if verbose:
                    l = [i - 1] + module.layers
                    sh = [list(x.shape)] + [list(out[i].shape) for i in module.layers]
                    str = '>>' + '+'.join(['layer %g %s' % x for x in zip(l, sh)])
                x = module(x,out)

            elif name == 'YOLOLayer':
                yolo_out.append(module(x))
            else:
                x = module(x)
            out.append(x if self.routs[i] else [])
            if verbose:
                print('%g/%g %s -' % (i,len(self.module_list),name),list(x.shape),str)
                str =''

        if self.training:
            return yolo_out
        else:
            x,p = zip(*yolo_out)
            x = torch.cat(x,1)
            return x,p

    def info(self,verbose=False):
        torch_utils.module_info(self,verbose)

def get_yolo_layers(self):

    return [i for i,m in enumerate(self.module_list) if m.__class__.__name__ == 'YOLOLayer']

