import math
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from build_utils.utils import xyxy2xywh, xywh2xyxy

help_url = 'https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data'
img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']

for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def exif_size(img):
    s = img.size
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:
            s = (s[1], s[0])
        elif rotation == 8:
            s = (s[1], s[0])
    except:
        pass

    return s


class LoadImagesAndLabels(Dataset):
    def __init__(self, path, img_size=416, batch_size=16, augment=False, hyp=None, rect=None,
                 cache_images=False, single_cls=False, pad=0.0, rank=-1):
        try:
            path = str(Path(path))
            if os.path.isfile(path):
                with open(path, 'r') as f:
                    f = f.read().splitlines()

            else:
                raise Exception('%s does not exist' % path)
            self.img_files = [x for x in f if os.path.splitext(x)[-1].lower() in img_formats]
        except Exception as e:
            raise FileNotFoundError("Error loading data from {}. {}".format(path, e))

        n = len(self.img_files)
        assert n > 0, "No images found in %s. See %s" % (path, help_url)

        bi = np.floor(np.arange(n) / batch_size).astype(np.int)
        nb = bi[-1] + 1

        self.n = n
        self.batch = bi
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.rect = rect
        self.mosaic = self.augment and not self.rect

        self.label_files = [x.replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt')
                            for x in self.img_files]
        sp = path.replace('.txt', '.shapes')
        try:
            with open(sp, 'r') as f:
                s = [x.split() for x in f.read().splitlines()]
                assert len(s) == n, "shapefile out of aync"
        except Exception as e:
            if rank in [-1, 0]:
                img_files = tqdm(self.img_files, desc='Reading image shapes')
            else:
                image_files = self.img_files
            s = [exif_size(Image.open('f')) for f in image_files]
            np.savetxt(sp, s, fmt='%g')

        self.shapes = np.array(s, dtype=np.float64)

        if self.rect:
            s = self.shapes
            ar = s[:, 1] / s[:, 0]
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.shapes = s[irect]
            ar = ar[irect]

            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()

                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / 32.0 + pad).astype(np.int) * 32

        self.imgs = [None] * n
        self.labels = [np.zeros((0, 5), dtype=np.float32)] * n
        extract_bounding_boxes, labels_loaded = False, False
        nm, nf, ne, nd = 0, 0, 0, 0
        if rect is True:
            np_labels_path = str(Path(self.label_files[0]).parent) + '.rect.npy'
        else:
            np_labels_path = str(Path(self.label_files[0]).parent) + 'norect.npy'

        if os.path.isfile(np_labels_path):
            x = np.load(np_labels_path, allow_pickle=True)
            if len(x) == n:
                self.labels = x
                labels_loaded = True

        if rank in [-1, 0]:
            pbar = tqdm(self.label_files)

        else:
            pbar = self.label_files

        for i, file in enumerate(pbar):
            if labels_loaded is True:
                l = self.labels[i]

            else:
                try:
                    with open(file, 'r') as f:
                        l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                except Exception as e:
                    print("An error occurred while loading the file {}: {}".format(file, e))
                    nm += 1
                    continue

            if l.shape[0]:
                assert l.shape[1] == 5, "> 5 label columns: %s" % file
                assert (l >= 0).all()
                assert (l[:, 1:] <= 1).all(), "non-normalized or out of bounds coordinate labels: %s" % file

                if np.unique(l, axis=0).shape[0] < l.shape[0]:
                    nd += 1
                if single_cls:
                    l[:, 0] = 0
                self.labels[i] = l
                nf += 1

                if extract_bounding_boxes:
                    p = Path(self.img_files[i])
                    img = cv2.imread(str(p))
                    h, w = img.shape[:2]
                    for j, x in enumerate(l):
                        f = '%s%sclassifier%s%g_%g_%s' % (p.parent.parent, os.sep, os.sep, x[0], j, p.name)
                        if not os.path.exists(Path(f).parent):
                            os.makedirs(Path(f).parent)

                        b = x[1:] * [w, h, w, h]
                        b[2:] = b[2:].max()
                        b[2:] = b[2:] * 1.3 + 30
                        b = xywh2xyxy(b.reshape(-1, 4)).revel().astype(np.int)

                        b[[0, 2]] = np.clip[b[[0, 2]], 0, w]
                        b[[1, 3]] = np.clip[b[[0, 2]], 0, h]
                        assert cv2.imwrite(f, img[b[1]:b[3], b[0]:b[2]]), "Failure extracting classifier boxes"
            else:
                ne += 1

            if rank in [-1, 0]:
                pbar.desc = "Caching labels (%g found, %g missing, %g empty, %g duplicate, for %g images)" % (
                    nf, nm, ne, nd, n)
        assert nf > 0, "No labels found in %s." % os.path.dirname(self.label_files[0]) + os.sep

        if not labels_loaded and n > 1000:
            print("Saving labels to %s for faster future loading" % np_labels_path)
            np.save(np_labels_path, self.labels)

        if cache_images:
            gb = 0
        if rank in [-1, 0]:
            pbar = tqdm(range(len(self.img_files), desc='Caching images'))
        else:
            pbar = range(len(self.img_files))

        self.img_hw0, self.img_hw0 = [None] * n, [None] * n
        for i in pbar:
            self.imgs[i], self.img_hw0[i], self.img_hw[i] = load_image(self, i)
        gb += self.imgs[i].nbytes
        if rank in [-1, 0]:
            pbar.desc = 'Caching images (%.1GB)' % (gb / 1E9)

        detect_corrupted_images = False
        if detect_corrupted_images:
            from skimage import io
            for file in tqdm(self.img_files, desc='Detecting corrupted images'):
                try:
                    _ = io.imread(file)
                except Exception as e:
                    print("Corrupted image detected: {}, {}".format(file, e))

        def __len__(self):
            return len(self.img_files)

        def __getitem__(self, index):
            hyp = self.hyp
            if self.mosaic:
                img, labels = load_mosaic(self, index)
                shapes = None
            else:
                img, (h0, w0), (h, w) = load_image(self, index)

                shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size
                img, ratio, pad = letterbox(img, shape, auto=False, scale_up=self.augment)
                shapes = (h0, w0), ((h / h0, w / w0), pad)

                labels = []
                x = self.labels[index]
                if x.size > 0:
                    labels = x.copy()
                    labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]
                    labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]
                    labels[:, 3] = ratio[0] * w * (x[:, 2] + x[:, 3] / 2) + pad[0]
                    labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]
