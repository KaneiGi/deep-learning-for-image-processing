import torch
from torch import nn, Tensor
from torch.jit.annotations import List

from .res50_backbone import resnet50
from .utils import dboxes300_coco, Encoder, PostProcess


class Backbone(nn.Module):
    def __init__(self, pretrain_path=None):
        super(Backbone, self).__init__()
        net = resnet50()
        self.out_channels = [1024, 512, 512, 256, 256, 256]

        if pretrain_path is not None:
            net.load_state_dict(torch.load(pretrain_path))

        self.feature_extractor = nn.Sequential(*list(net.children())[:7])

        conv4_block1 = self.feature_extractor[-1][0]
        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0] = (1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


class SSD300(nn.Module):
    def __init__(self, backbone=None, num_classes=21):
        super(SSD300, self).__init__()

        if backbone is None:
            raise Exception('backbone is None')
        if hasattr(backbone, 'out_channels'):
            raise Exception('the backbone has no attributes: out_channels')
        self.features_extractor = backbone

        self.num_classes = num_classes

        self._bulid_additional_features(self.features_extractor.out_channels)
        self.num_defaults = [4, 6, 6, 6, 4, 4]
        location_extractors = []
        confidence_extractors = []

        for nd, oc in zip(self.num_defaults, self.features_extractor):
            location_extractors.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
            confidence_extractors.append(nn.Conv2d(oc, nd * self.num_classes, kernel_size=3, padding=1))

        self.loc = nn.ModuleList(
            location_extractors)  # modulelist用法和list相同，区别在于modulelist中添加子module以后，其参数也会自动注册到整个list中
        self.conf = nn.ModuleList(confidence_extractors)  # 但是不能实现sequence那样的自动向前传播过程，需要用for循环遍历

        self._init_weights()

        default_box = dboxes300_coco()
        self.compute_loss = Loss()
        self.encoder = Encoder(default_box)
        self.postprocess = PostProcess(default_box)

    def _build_additional_features(self, input_size):

        additional_blocks = []

        middle_channels = [256, 256, 128, 128, 128]
        for i, (input_channel, output_channel, middle_channel) in enumerate(
                zip(input_size[:-1], input_size[1:], middle_channels)):
            padding, stride = (1, 2) if i < 3 else (0, 1)
            layer = nn.Sequential(
                nn.Conv2d(input_channel, middle_channel, kernel_size=1, bias=False),
                nn.BatchNorm2d(middle_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(middle_channel, output_channel, kernel_size=3, padding=padding, stride=stride, bias=False),
                nn.BatchNorm2d(output_channel),
                nn.ReLU(inplace=True)
            )
            additional_blocks.append(layer)
        self.additional_blocks = nn.ModuleList(additional_blocks)

    def _init_weights(self):
        layers = [*self.additional_blocks, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def bbox_view(self, features, loc_extractor, conf_extractor):
        locs = []
        confs = []

        for feature, location, confidence in zip(features, loc_extractor, conf_extractor):
            locs.append(location(feature).view(feature.size(0), 4, -1))
            confs.append(confidence(feature).view(feature.size(0), self.num_classes, -1))

        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()
        return locs, confs

    def forward(self, image, targets=None):

        x = self.features_extractor(image)

        detection_features = torch.jit.annotate(List[Tensor], [])
        detection_features.append(x)
        for layer in self.additional_blocks:
            x = layer(x)
            detection_features.append(x)

        locs, confs = self.bbox_view(detection_features, self.loc, self.conf)

        if self.training:
            if targets is None:
                raise Exception('In train mode, targets should be passed')
            bboxes_out = targets['boxes']
            bboxes_out = bboxes_out.transpose(1, 2).contiguous()
            labels_out = targets['labels']

            loss = self.compute_loss(locs, confs, bboxes_out, labels_out)
            return {'total_losses': loss}
        results = self.postprocess(locs, confs)
        return results


class Loss(nn.Module):
    def __init__(self, dboxes):
        super(Loss, self).__init__()
        self.scale_xy = 1.0 / dboxes.scale_xy
        self.scale_wh = 1.0 / dboxes.scale_wh

        self.location_loss = nn.SmoothL1Loss(reduction='none')
        self.confidence_loss = nn.CrossEntropyLoss(reduction='none')

    def _location_vec(self, loc):
        gxy = self.scale_xy * (loc[:, :2, :] - self.dboxes[:, :2, :]) / self.dboxes[:, 2:, :]
        gwh = self.scale_wh * (loc[:, :2, :] - self.dboxes[:, 2:, :]) / self.dboxes[:, 2:, :]
        return torch.cat((gxy, gwh), dim=1).contiguous()

    def forward(self, predict_locs, predict_confs, ground_locs, ground_lables):
        pos_mask = torch.gt(ground_lables, 0)
        pos_num = pos_mask.sum(dim=1)
        vec_gd = self._location_vec(ground_locs)

        loc_loss = self.location_loss(predict_locs, vec_gd).sum(dim=1)
        pos_loc_loss = (pos_mask.float() * loc_loss).sum(dim=1)

        total_conf_loss = self.confidence_loss(predict_confs, ground_lables)

        con_neg = total_conf_loss.clone()
        con_neg[pos_mask] = 0.0

        _, con_idx = con_neg.sort(dim=1, descending=True)
        _, con_rank = con_neg.sort(dim=1)

        neg_sum = torch.clamp(pos_num * 3, max=pos_mask.size(0)).unsqueeze(-1)
        neg_mask = torch.lt(con_rank, neg_sum)

        conf_loss = (total_conf_loss * (neg_mask.float() + pos_mask.float()))
        total_loss = loc_loss + conf_loss

        num_mask = torch.gt(pos_num, 0).float()
        pos_num = pos_num.float().clamp(min=1e-6)

        result = (total_loss * num_mask / pos_num).mean(dim=0)
        return result
