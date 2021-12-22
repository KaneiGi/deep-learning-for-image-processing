from math import sqrt
import itertools

import torch
import torch.nn.functional as F
from torch.jit.annotations import Tuple, List
from torch import nn, Tensor
import numpy as np


def box_area(boxes):
    return (boxes[:, 2] - boxes[0]) * (boxes[:, 3] - boxes[:, 1])


def cal_iou_tensor(boxes_1, boxes_2):
    area_1 = box_area(boxes_1)
    area_2 = box_area(boxes_2)

    lt = torch.max(boxes_1[:, None, :2], boxes_2[:, :2])
    rb = torch.min(boxes_1[:, None, 2:], boxes_2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :0] * wh[:, :, 1]

    iou = inter / (area_1[:, None] + area_2 - inter)
    return iou


class Encoder(object):

    def __init__(self, dboxes):
        self.dboxes = dboxes(order='ltrb')
        self.dboxes_xywh = dboxes(order='xywh').unsqueeze(dim=0)
        self.num_boxes = dboxes.size(0)
        self.scale_xy = dboxes.scale_xy
        self.scale_wh = dboxes.scale_wh

    def encode(self, bboxes_in, labels_in, criteria=0.5):
        ious = cal_iou_tensor(bboxes_in, self.dboxes)
        best_dbox_ious, best_dbox_idx = ious.max(dim=0)
        best_bbox_ious, best_bbox_idx = ious.max(dim=1)

        best_dbox_ious.index_fill_(0, best_bbox_idx, 2.0)

        idx = torch.arange(0, best_bbox_idx.size(0), dtype=torch.int64)
        best_dbox_idx[best_bbox_idx[idx]] = idx

        masks = best_dbox_ious > criteria

        labels_out = torch.zeros(self.num_boxes, dtype=torch.int64)
        labels_out[masks] = labels_in[best_dbox_idx[masks]]

        bboxes_out = self.dboxes.clone()
        bboxes_out[masks, :] = bboxes_in[best_dbox_idx[masks], :]

        x = 0.5 * (bboxes_out[:, 0] + bboxes_out[:, 2])
        y = 0.5 * (bboxes_out[:, 1] + bboxes_out[:, 3])
        w = bboxes_out[:2] - bboxes_out[:0]
        h = bboxes_out[:3] - bboxes_out[:1]
        bboxes_out[:, 0] = x
        bboxes_out[:, 1] = y
        bboxes_out[:, 2] = w
        bboxes_out[:, 3] = h

        return bboxes_out, labels_out
