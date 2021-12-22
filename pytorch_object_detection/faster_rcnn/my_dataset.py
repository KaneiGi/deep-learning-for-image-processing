import math
import platform
import random

import cv2
from torch.utils.data import Dataset
import os
import torch
import json
from PIL import Image,ImageFont,ImageDraw
from lxml import etree
import numpy as np
import torchvision.transforms as transforms


def calc_iou_tensor(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    #  When the shapes do not match,
    #  the shape of the returned output tensor follows the broadcasting rules
    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # left-top [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom [N,M,2]

    wh = (rb - lt).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def random_affine1(img, boxes=(), labels=(), iscrowd=(), degrees=10, translate=.1, scale=.1, shear=10, border=0):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
    # boxes = [cls, xyxy]
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if not isinstance(boxes, np.ndarray):
        boxes = np.array(boxes)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    if not isinstance(iscrowd, np.ndarray):
        iscrowd = np.array(iscrowd)
    if type(border) == int:
        border = [border,border]
    # 给定的输入图像的尺寸(416/512/640)，如果是在mosaic里面调用的random_affine，那么输入图像的尺寸会比指定的self.img_size大两倍，为了还原尺寸，这里border=-img_size/2
    height = img.shape[0] - border[0]
    width = img.shape[1] - border[1]

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)  # uniform() 方法将随机生成下一个实数，它在 [x, y] 范围内
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[1]   # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[0]   # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    # *和np.multiply只能做点乘运算，当运算符两边的数据维度无法满足点乘运算结果时，就会报错
    # @和.dot只能做矩阵乘法运算
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border != 0) or (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(boxes)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1 ,左上角，右下角，左下角，右上角
        # [4*n, 3] -> [n, 8]
        xy = (xy @ M.T)[:, :2].reshape(n, 8)  # 确实要M的转置才行，因为原本的顺序是M@xy(3x3@3x1 = 3x1),现在是xy.T的形式，相应的M.T才行
        # 等价于(M.T @ xy.T).T
        # create new boxes
        x = xy[:, [0, 2, 4, 6]]  # [n, 4]
        y = xy[:, [1, 3, 5, 7]]  # [n, 4]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4,
                                                                              n).T  # [n, 4]，因为最终的box是没有倾斜的，这样就会形成一个外接的矩形包住原来的倾斜box

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # reject warped points outside of image
        # 对坐标进行裁剪，防止越界
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]

        # 计算调整后的每个box的面积
        area = w * h
        # 计算调整前的每个box的面积
        area0 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        # 计算每个box的比例
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        # 选取长宽大于4个像素，且调整前后面积比例大于0.2，且比例小于10的box,那些部分出界的box也会被移除,这里0.2太小了
        i = (w > 4) & (h > 4) & (area / (area0 * s + 1e-16) > 0.8) & (ar < 10)

        boxes = boxes[i]  # i 在这里是一个模板，过滤掉经过处理后不符合的target_box
        labels = labels[i]
        iscrowd = iscrowd[i]
        boxes[:, :] = xy[i]

    return img, boxes, labels, iscrowd

def random_affine(img, boxes=(), labels=(), iscrowd=(), degrees=10, translate=.1, scale=.1, shear=10, border=0):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
    # boxes = [cls, xyxy]
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if not isinstance(boxes, np.ndarray):
        boxes = np.array(boxes)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    if not isinstance(iscrowd, np.ndarray):
        iscrowd = np.array(iscrowd)

    # 给定的输入图像的尺寸(416/512/640)，如果是在mosaic里面调用的random_affine，那么输入图像的尺寸会比指定的self.img_size大两倍，为了还原尺寸，这里border=-img_size/2
    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)  # uniform() 方法将随机生成下一个实数，它在 [x, y] 范围内
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    # *和np.multiply只能做点乘运算，当运算符两边的数据维度无法满足点乘运算结果时，就会报错
    # @和.dot只能做矩阵乘法运算
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border != 0) or (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(boxes)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1 ,左上角，右下角，左下角，右上角
        # [4*n, 3] -> [n, 8]
        xy = (xy @ M.T)[:, :2].reshape(n, 8)  # 确实要M的转置才行，因为原本的顺序是M@xy(3x3@3x1 = 3x1),现在是xy.T的形式，相应的M.T才行
        # 等价于(M.T @ xy.T).T
        # create new boxes
        x = xy[:, [0, 2, 4, 6]]  # [n, 4]
        y = xy[:, [1, 3, 5, 7]]  # [n, 4]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4,
                                                                              n).T  # [n, 4]，因为最终的box是没有倾斜的，这样就会形成一个外接的矩形包住原来的倾斜box

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # reject warped points outside of image
        # 对坐标进行裁剪，防止越界
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]

        # 计算调整后的每个box的面积
        area = w * h
        # 计算调整前的每个box的面积
        area0 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        # 计算每个box的比例
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        # 选取长宽大于4个像素，且调整前后面积比例大于0.2，且比例小于10的box,那些部分出界的box也会被移除
        i = (w > 4) & (h > 4) & (area / (area0 * s + 1e-16) > 0.8) & (ar < 10)

        boxes = boxes[i]  # i 在这里是一个模板，过滤掉经过处理后不符合的target_box
        labels = labels[i]
        iscrowd = iscrowd[i]
        boxes[:, :] = xy[i]

    return img, boxes, labels, iscrowd
def random_crop(img, boxes, labels, iscrowd):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if not isinstance(boxes, np.ndarray):
        boxes = np.array(boxes)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    if not isinstance(iscrowd, np.ndarray):
        iscrowd = np.array(iscrowd)
    h0, w0 = img.shape[:2]
    sample_options = (
        # Do nothing
        None,
        # min IoU, max IoU
        (0.1, None),
        (0.3, None),
        (0.5, None),
        (0.7, None),
        (0.9, None),
        # no IoU requirements
        (None, None),
    )
    while True:
        mode = random.choice(sample_options)
        if mode is None:
            return img, boxes, labels, iscrowd

        min_iou, max_iou = mode
        min_iou = float('-inf') if min_iou is None else min_iou
        max_iou = float('+inf') if max_iou is None else max_iou

        for _ in range(5):
            w = random.uniform(0.3 * w0, w0)
            h = random.uniform(0.3 * h0, h0)

            if w / h < 0.5 or w / h > 2:  # 保证宽高比例在0.5-2之间
                continue

            # left 0 ~ wtot - w, top 0 ~ htot - h
            left = random.uniform(0, w0 - w)
            top = random.uniform(0, h0 - h)

            right = left + w
            bottom = top + h

            ious = calc_iou_tensor(boxes, np.array([[left, top, right, bottom]]))
            if not ((ious > min_iou) & (ious < max_iou)).all():
                continue

            xc = 0.5 * (boxes[:, 0] + boxes[:, 2])
            yc = 0.5 * (boxes[:, 1] + boxes[:, 3])

            # 查找所有的gt box的中心点有没有在采样patch中的
            masks = (xc > left) & (xc < right) & (yc > top) & (yc < bottom)

            # if no such boxes, continue searching again
            # 如果所有的gt box的中心点都不在采样的patch中，则重新找
            if not masks.any():
                continue
            boxes[boxes[:, 0] < left, 0] = left
            boxes[boxes[:, 1] < top, 1] = top
            boxes[boxes[:, 2] > right, 2] = right
            boxes[boxes[:, 3] > bottom, 3] = bottom

            boxes = boxes[masks]
            labels = labels[masks]
            iscrowd = iscrowd[masks]

            boxes[:, 0] = (boxes[:, 0] - left)
            boxes[:, 1] = (boxes[:, 1] - top)
            boxes[:, 2] = (boxes[:, 2] - left)
            boxes[:, 3] = (boxes[:, 3] - top)

            img = img[int(top):int(bottom), int(left):int(right), :]

            return img, boxes, labels, iscrowd


class VOC2012DataSet(Dataset):
    """读取解析PASCAL VOC2012数据集"""

    def __init__(self, voc_root='C:', transforms=None, txt_name: str = "train.txt", augment=True):
        self.augment = augment
        if platform.system() in [ "Darwin",'Windows']:
            self.root = voc_root
            self.img_root = os.path.join(self.root, "images")
            self.annotations_root = os.path.join(self.root, "annotations")
            txt_path = os.path.join(self.root, txt_name)

        # else:
        #     self.root = os.path.join(voc_root, "VOCdevkit", "VOC2012")
        #     self.img_root = os.path.join(self.root, "JPEGImages")
        #     self.annotations_root = os.path.join(self.root, "Annotations")
        #     txt_path = os.path.join(self.root, "ImageSets", "Main", txt_name)

        # read train.txt or val.txt file

        assert os.path.exists(txt_path), "not found {} file.".format(txt_name)

        with open(txt_path,encoding='utf-8') as read:
            self.xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml")
                             for line in read.readlines()]

        # check file
        assert len(self.xml_list) > 0, "in '{}' file does not find any information.".format(txt_path)
        for xml_path in self.xml_list:
            assert os.path.exists(xml_path), "not found '{}' file.".format(xml_path)

        # read class_indict
        if platform.system() == "Darwin":
            json_file = r"/Users/gi/OneDrive/data_set/classes.json"
        else:
            json_file = r"C:\Users\wei43\OneDrive\data_set\classes.json"
        assert os.path.exists(json_file), "{} file not exist.".format(json_file)
        json_file = open(json_file, 'r',encoding='utf-8')
        self.class_dict = json.load(json_file)
        self.class_dict_inverse = {}
        for key,value in self.class_dict.items():
           self.class_dict_inverse[str(value)] = key

        self.transforms = transforms

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx):

        self.mosaic = False
        self.crop = False
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path,encoding='utf-8') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        img_path = os.path.join(self.img_root, data["filename"])

        if img_path.split('.')[-1].upper() not in ["JPG", "PNG"]:
            raise ValueError("Image '{}' format not JPEG or PNG".format(img_path))
        img = cv2.imread(img_path)

        boxes = []
        labels = []
        iscrowd = []

        assert "object" in data, "{} lack of object information.".format(xml_path)
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])

            # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
            if xmax <= xmin or ymax <= ymin:
                print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)
        # for box in boxes:
        #     for i,_ in enumerate(box):
        #         box[i] = int(box[i])
        #     img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 10)
        # cv2.namedWindow('img_original', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('img_original', 1080, 720)
        # cv2.imshow("img_original", img)
        # cv2.waitKey(0)
        # cv2.destroyWindow('img_original')

        # convert everything into a torch.Tensor
        # self.mosaic = False
        if self.augment:

            if random.random() < 0.0:
                self.crop = True
                img, boxes, labels, iscrowd = random_crop(img, boxes, labels, iscrowd)
                # for box in boxes.astype(np.int64):
                #     img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 10)
                # cv2.namedWindow('img_crop',cv2.WINDOW_NORMAL)
                # cv2.resizeWindow('img_crop',1080,720)
                # cv2.imshow("img_crop", img)
                # cv2.waitKey(0)
                # cv2.destroyWindow('img_crop')

            if random.random() < 1 and not self.crop:  # random mosaic
                self.mosaic = True
                boxes = np.zeros((0, 4))
                labels = []
                iscrowd = []
                indices = [idx] + [random.randint(0, len(self.xml_list) - 1) for _ in range(3)]
                for i, index in enumerate(indices):
                    xml_path = self.xml_list[index]
                    with open(xml_path,encoding='utf-8') as fid:
                        xml_str = fid.read()
                    xml = etree.fromstring(xml_str)
                    data = self.parse_xml_to_dict(xml)["annotation"]
                    img_path = os.path.join(self.img_root, data["filename"])
                    img = cv2.imread(img_path)
                    # h0,w0 = (1080,1920)
                    h0,w0 = img.shape[:2]
                    interp = cv2.INTER_AREA
                    img = cv2.resize(img,(int(w0/2),int(h0/2)),interpolation=interp)
                    h, w = img.shape[:2]

                    if i == 0:  # top left
                        # 创建马赛克图像
                        # h4, w4 = img.shape[0] * 2, img.shape[1] * 2
                        h4,w4 = (1080,1920)
                        img4 = np.full((h4, w4, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                        # cv2.imshow('img4', img4)
                        # cv2.waitKey(1)
                        # cv2.destroyWindow('img4')

                        # yc, xc = [int(random.uniform(size * 0.5, size * 1.5)) for size in img.shape[:2]]
                        yc,xc = int(h4/2),int(w4/2)
                        # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
                        x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h,
                                                                 0), xc, yc  # xmin, ymin, xmax, ymax (large image),img4的索引
                        # 计算截取的图像区域信息(以xc,yc为第一张图像的右下角坐标填充到马赛克图像中，丢弃越界的区域)，img的索引
                        x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (
                                y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
                    elif i == 1:  # top right
                        # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
                        x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, w4), yc
                        # 计算截取的图像区域信息(以xc,yc为第二张图像的左下角坐标填充到马赛克图像中，丢弃越界的区域)
                        x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
                    elif i == 2:  # bottom left
                        # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
                        x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(h4, yc + h)
                        # 计算截取的图像区域信息(以xc,yc为第三张图像的右上角坐标填充到马赛克图像中，丢弃越界的区域)
                        x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
                    elif i == 3:  # bottom right
                        # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
                        x1a, y1a, x2a, y2a = xc, yc, min(xc + w, w4), min(h4, yc + h)
                        # 计算截取的图像区域信息(以xc,yc为第四张图像的左上角坐标填充到马赛克图像中，丢弃越界的区域)
                        x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
                    # ya_diff, xa_diff, yb_diff, xb_diff = y2a - y1a, x2a - x1a, y2b - y1b, x2b - x1b
                    # 将截取的图像区域填充到马赛克图像的相应位置
                    # assert 0 <= y1a and h4 >= y1a, 'failure'
                    # assert 0 <= y2a and h4 >= y2a, 'failure'
                    # assert 0 <= x1a and w4 >= x1a, 'failure'
                    # assert 0 <= x2a and w4 >= x2a, 'failure'
                    # assert 0 <= y1b and h >= y1b, 'failure'
                    # assert 0 <= y2b and h >= y2b, 'failure'
                    # assert 0 <= x1b and w >= x1b, 'failure'
                    # assert 0 <= x2b and w >= x2b, 'failure'

                    img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
                    # cv2.imshow('img4', img4)
                    # cv2.waitKey(0)
                    # cv2.destroyWindow('img4')
                    # 计算pad(图像边界与马赛克边界的距离，越界的情况为负值)
                    padw = x1a - x1b
                    padh = y1a - y1b

                    for obj in data['object']:
                        xmin = float(obj['bndbox']['xmin'])/2 + padw
                        xmax = float(obj['bndbox']['xmax'])/2 + padw
                        ymin = float(obj['bndbox']['ymin'])/2 + padh
                        ymax = float(obj['bndbox']['ymax'])/2 + padh
                        xca = (xmin + xmax) / 2
                        yca = (ymin + ymax) / 2
                        if (0 > xca or w4 < xca) or (0 > yca or h4 < yca):
                            # print(f"Warning: deleting {xml_path}'s box out ")
                            continue

                        if xmax <= xmin or ymax <= ymin:
                            print('Warning: in {} xml, there are some bndbox w/h <=0'.format(xml_path))
                            continue
                        arr = np.array([xmin, ymin, xmax, ymax])
                        arr[[0, 2]] = arr[[0, 2]].clip(0, w4)
                        arr[[1, 3]] = arr[[1, 3]].clip(0, h4)

                        boxes = np.r_[boxes, arr.reshape(-1, 4)]
                        labels.append(self.class_dict[obj['name']])
                        if 'difficult' in obj:
                            iscrowd.append(int(obj['difficult']))
                        else:
                            iscrowd.append(0)

                # Augment
                # img4 = img4[s // 2: int(s * 1.5), s // 2:int(s * 1.5)]  # center crop (WARNING, requires box pruning)
                # cv2.namedWindow('img4',0)
                # cv2.resizeWindow('img4',1080,720)
                # for box in boxes.astype(np.int64):
                #     img4 = cv2.rectangle(img4,(box[0],box[1]),(box[2],box[3]),(0,255,0),2)
                # cv2.imshow("img4", img4)
                # cv2.waitKey(0)
                # cv2.destroyWindow('img4')
                img = img4
                # img, boxes,labels,iscrowd = random_affine(img4, boxes,labels,iscrowd)  # border to remove
                # img, boxes,labels,iscrowd = random_affine(img4, boxes,labels,iscrowd,border=(h,w))  # border to remove



                # for box in boxes.astype(np.int64):
                #     img = cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(255,0,0),10)
                # cv2.namedWindow('img_masaic',cv2.WINDOW_NORMAL)
                # cv2.resizeWindow('img_masaic',1080,720)
                # cv2.imshow("img_masaic", img)
                # cv2.waitKey(0)
                # cv2.destroyWindow('img_mosaic')

            # boxes = np.array(boxes)
            # img = np.array(image)
            if not self.mosaic:
                if random.random()<0.5:
                    # img, boxes, labels, iscrowd = random_affine(img, boxes, labels, iscrowd)
                    pass
                # boxes = np.array(boxes)
                # for box in boxes.astype(np.int64):
                #     img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 10)
                # cv2.namedWindow('img_final',cv2.WINDOW_NORMAL)
                # cv2.resizeWindow('img_final',1080,720)
                # cv2.imshow("img_final", img)
                # cv2.waitKey(0)
                # cv2.destroyWindow('img_final')

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = np.ascontiguousarray(img)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        import matplotlib.pyplot as plt
        # img_plt = transforms.ToPILImage()(img).convert('RGB')
        # img_plt.show()


        # img_cv = img.numpy()*255
        # img_cv = img_cv.astype(np.uint8)
        # img_cv = np.ascontiguousarray(np.transpose(img_cv,(1,2,0)))
        # for box in target['boxes'].numpy().astype(np.int64):
        #     img_cv = cv2.rectangle(img_cv, (box[0], box[1]), (box[2], box[3]), (255, 0, 255), 10)
        # cv2.namedWindow('img_cv', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('img_cv', 1080, 720)
        # cv2.imshow('img_cv',img_cv)
        # cv2.waitKey(0)
        # cv2.destroyWindow('img_cv')



        # pil_img = (img.numpy()*255 ).astype(np.uint8)
        # pil_img = np.ascontiguousarray(np.transpose(pil_img,(1,2,0)))
        # pil_img = Image.fromarray(pil_img)
        # draw = ImageDraw.Draw(pil_img)
        # if platform.system() == 'Windows':
        #     font = ImageFont.truetype(r'C:\WINDOWS\Fonts\MSGOTHIC.ttc', 40)
        # for box,label in zip(target['boxes'].numpy(),target['labels'].numpy()):
        #     draw.text((box[0],box[1]-60),self.class_dict_inverse[str(label)],fill='red',font=font)
        #     draw.rectangle((box[0],box[1],box[2],box[3]),outline='red',width=10)
        # pil_img.show()

        return img, target

    def __getitem0__(self, idx):

        self.mosaic = False
        self.crop = False
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path,encoding='utf-8') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        # img_path = os.path.join(self.img_root, data["filename"])
        img_path = os.path.join(self.root, data["folder"], data["filename"])

        if img_path.split('.')[-1].upper() not in ["JPG", "PNG"]:
            raise ValueError("Image '{}' format not JPEG or PNG".format(img_path))
        img = cv2.imread(img_path)
        boxes = []
        labels = []
        iscrowd = []

        assert "object" in data, "{} lack of object information.".format(xml_path)
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])

            # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
            if xmax <= xmin or ymax <= ymin:
                print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)

        # convert everything into a torch.Tensor
        # self.mosaic = False
        if self.augment:

            if random.random() < 0.0:
                self.crop = True
                img, boxes, labels, iscrowd = random_crop(img, boxes, labels, iscrowd)
                # for box in boxes.astype(np.int64):
                #     img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                # cv2.imshow("img_crop", img)
                # cv2.waitKey(0)
                # cv2.destroyWindow('img_crop')

            if random.random() < 0.5 and not self.crop:  # random mosaic
                self.mosaic = True
                boxes = np.zeros((0, 4))
                labels = []
                iscrowd = []
                indices = [idx] + [random.randint(0, len(self.xml_list) - 1) for _ in range(3)]
                for i, index in enumerate(indices):
                    xml_path = self.xml_list[index]
                    with open(xml_path,encoding='utf-8') as fid:
                        xml_str = fid.read()
                    xml = etree.fromstring(xml_str)
                    data = self.parse_xml_to_dict(xml)["annotation"]
                    # img_path = os.path.join(self.img_root, data["filename"])
                    img_path = os.path.join(self.root, data["folder"], data["filename"])
                    img = cv2.imread(img_path)
                    h0, w0 = img.shape[:2]
                    interp = cv2.INTER_AREA
                    img = cv2.resize(img, (int(w0 / 2), int(h0 / 2)), interpolation=interp)
                    h, w = img.shape[:2]

                    if i == 0:  # top left
                        # 创建马赛克图像
                        h4, w4 = img.shape[0] * 2, img.shape[1] * 2
                        img4 = np.full((h4, w4, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                        # cv2.imshow('img4', img4)
                        # cv2.waitKey(1)
                        # cv2.destroyWindow('img4')

                        yc, xc = [int(random.uniform(size * 0.5, size * 1.5)) for size in img.shape[:2]]
                        # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
                        x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h,
                                                                 0), xc, yc  # xmin, ymin, xmax, ymax (large image),img4的索引
                        # 计算截取的图像区域信息(以xc,yc为第一张图像的右下角坐标填充到马赛克图像中，丢弃越界的区域)，img的索引
                        x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (
                                y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
                    elif i == 1:  # top right
                        # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
                        x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, w4), yc
                        # 计算截取的图像区域信息(以xc,yc为第二张图像的左下角坐标填充到马赛克图像中，丢弃越界的区域)
                        x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
                    elif i == 2:  # bottom left
                        # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
                        x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(h4, yc + h)
                        # 计算截取的图像区域信息(以xc,yc为第三张图像的右上角坐标填充到马赛克图像中，丢弃越界的区域)
                        x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
                    elif i == 3:  # bottom right
                        # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
                        x1a, y1a, x2a, y2a = xc, yc, min(xc + w, w4), min(h4, yc + h)
                        # 计算截取的图像区域信息(以xc,yc为第四张图像的左上角坐标填充到马赛克图像中，丢弃越界的区域)
                        x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
                    # ya_diff, xa_diff, yb_diff, xb_diff = y2a - y1a, x2a - x1a, y2b - y1b, x2b - x1b
                    # 将截取的图像区域填充到马赛克图像的相应位置
                    # assert 0 <= y1a and h4 >= y1a, 'failure'
                    # assert 0 <= y2a and h4 >= y2a, 'failure'
                    # assert 0 <= x1a and w4 >= x1a, 'failure'
                    # assert 0 <= x2a and w4 >= x2a, 'failure'
                    # assert 0 <= y1b and h >= y1b, 'failure'
                    # assert 0 <= y2b and h >= y2b, 'failure'
                    # assert 0 <= x1b and w >= x1b, 'failure'
                    # assert 0 <= x2b and w >= x2b, 'failure'

                    img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
                    # cv2.imshow('img4', img4)
                    # cv2.waitKey(1)
                    # cv2.destroyWindow('img4')
                    # 计算pad(图像边界与马赛克边界的距离，越界的情况为负值)
                    padw = x1a - x1b
                    padh = y1a - y1b

                    for obj in data['object']:
                        xmin = float(obj['bndbox']['xmin']) / 2 + padw
                        xmax = float(obj['bndbox']['xmax']) / 2 + padw
                        ymin = float(obj['bndbox']['ymin']) / 2 + padh
                        ymax = float(obj['bndbox']['ymax']) / 2 + padh
                        xca = (xmin + xmax) / 2
                        yca = (ymin + ymax) / 2
                        if (0 > xca or w4 < xca) or (0 > yca or h4 < yca):
                            # print(f"Warning: deleting {xml_path}'s box out ")
                            continue

                        if xmax <= xmin or ymax <= ymin:
                            print('Warning: in {} xml, there are some bndbox w/h <=0'.format(xml_path))
                            continue
                        arr = np.array([xmin, ymin, xmax, ymax])
                        arr[[0, 2]] = arr[[0, 2]].clip(0, w4)
                        arr[[1, 3]] = arr[[1, 3]].clip(0, h4)

                        boxes = np.r_[boxes, arr.reshape(-1, 4)]
                        labels.append(self.class_dict[obj['name']])
                        if 'difficult' in obj:
                            iscrowd.append(int(obj['difficult']))
                        else:
                            iscrowd.append(0)

                # Augment
                # img4 = img4[s // 2: int(s * 1.5), s // 2:int(s * 1.5)]  # center crop (WARNING, requires box pruning)
                # for box in boxes.astype(np.int64):
                #     img4 = cv2.rectangle(img4,(box[0],box[1]),(box[2],box[3]),(0,255,0),2)
                #     cv2.imshow("img4", img4)
                #     cv2.waitKey(0)
                #     cv2.destroyWindow('img4')

                # for box in boxes.astype(np.int64):
                #     img4 = cv2.rectangle(img4,(box[0],box[1]),(box[2],box[3]),(0,255,0),2)
                # cv2.imshow("img4", img4)
                # cv2.waitKey(0)
                # cv2.destroyWindow('img4')
                # img = img4
                img, boxes,labels,iscrowd = random_affine(img4, boxes,labels,iscrowd)  # border to remove
                for box in boxes.astype(np.int64):
                    img = cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(255,0,0),2)
                cv2.imshow("img_masaic", img)
                cv2.waitKey(0)
                cv2.destroyWindow('img_masaic')

            # boxes = np.array(boxes)
            # img = np.array(image)
            # if not self.mosaic:
            # if random.random()<0.5:
            #     img, boxes, labels, iscrowd = random_affine(img, boxes, labels, iscrowd)
            # boxes = np.array(boxes)
            # for box in boxes.astype(np.int64):
            #     img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            # cv2.imshow("img_final", img)
            # cv2.waitKey(0)
            # cv2.destroyWindow('img_final')

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.ascontiguousarray(img)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
    def get_height_and_width(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path,encoding='utf-8') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

    def parse_xml_to_dict(self, xml):
        """
        将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
        Args:
            xml: xml tree obtained by parsing XML file contents using lxml.etree

        Returns:
            Python dictionary holding XML contents.
        """

        if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    def coco_index(self, idx):
        """
        该方法是专门为pycocotools统计标签信息准备，不对图像和标签作任何处理
        由于不用去读取图片，可大幅缩减统计时间

        Args:
            idx: 输入需要获取图像的索引
        """
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path,encoding='utf-8') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        # img_path = os.path.join(self.img_root, data["filename"])
        # image = Image.open(img_path)
        # if image.format != "JPEG":
        #     raise ValueError("Image format not JPEG")
        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            iscrowd.append(int(obj["difficult"]))

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return (data_height, data_width), target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


# import transforms
# from draw_box_utils import draw_box
# from PIL import Image
# import json
# import matplotlib.pyplot as plt
# import torchvision.transforms as ts
# import random
#
# # read class_indict
# category_index = {}
# try:
#     json_file = open('./pascal_voc_classes.json', 'r')
#     class_dict = json.load(json_file)
#     category_index = {v: k for k, v in class_dict.items()}
# except Exception as e:
#     print(e)
#     exit(-1)
#
# data_transform = {
#     "train": transforms.Compose([transforms.ToTensor(),
#                                  transforms.RandomHorizontalFlip(0.5)]),
#     "val": transforms.Compose([transforms.ToTensor()])
# }
#
# # load train data set
# train_data_set = VOC2012DataSet(os.getcwd(), data_transform["train"], "train.txt")
# print(len(train_data_set))
# for index in random.sample(range(0, len(train_data_set)), k=5):
#     img, target = train_data_set[index]
#     img = ts.ToPILImage()(img)
#     draw_box(img,
#              target["boxes"].numpy(),
#              target["labels"].numpy(),
#              [1 for i in range(len(target["labels"].numpy()))],
#              category_index,
#              thresh=0.5,
#              line_thickness=5)
#     plt.imshow(img)
#     plt.show()
if __name__ == '__main__':
    # xml_path = 'C:/Users/wei43/Downloads/VOC2012/Annotations/2007_000027.xml'
    # dataset = VOC2012DataSet()
    # with open(xml_path) as fid:
    #     xml_str = fid.read()
    # xml = etree.fromstring(xml_str)
    # data = dataset.parse_xml_to_dict(xml)["annotation"]
    b = 0
    a = {b: "0"}
    print(a)
