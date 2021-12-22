import math
from typing import List, Tuple, Dict, Optional

import torch
from torch import nn, Tensor
import torchvision

# import sys,os
# sys.path.append(os.getcwd())
from .image_list import ImageList


# from  torch.onnx import operators
# image = torch.zeros((3,3,3))
# size = torch.onnx.operators.shape_as_tensor(image)[-2:]
# size_1 = image.shape[-2:]
# size_2 = image.size()[-2:]
# size_2 = torch._shape_as_tensor(image)[-2:]
# print(size,size_1,size_2)

@torch.jit.unused
def _resize_image_onnx(image, self_min_size, self_max_size):
    # type:(Tensor,float,float)->Tensor
    from torch.onnx import operators
    im_shape = operators.shape_as_tensor(image)[-2:]
    min_size = torch.min(im_shape).to(dtype=torch.float32)
    max_size = torch.max(im_shape).to(dtype=torch.float32)
    scale_factor = torch.min(self_min_size / min_size, self_max_size / max_size)

    image = torch.nn.functional.interploate(
        image[None], scale_factor=scale_factor, mode='bilinear', recompute_scale_factor=True,
        align_corners=False)[0]

    return image


def _resize_image(image, self_min_size, self_max_size):
    # type:(Tensor,float,float)->Tensor

    im_shape = torch.tensor((image)[-2:])
    min_size = torch.min(im_shape).to(dtype=torch.float32)
    max_size = torch.max(im_shape).to(dtype=torch.float32)
    scale_factor = torch.min(self_min_size / min_size, self_max_size / max_size)

    image = torch.nn.functional.interploate(
        image[None], scale_factor=scale_factor, mode='bilinear', recompute_scale_factor=True,
        align_corners=False)[0]

    return image


def resize_boxes(boxes, original_size, new_size):
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratios_height, ratios_width = ratios

    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratios_width
    xmax = xmax * ratios_width
    ymin = ymin * ratios_height
    ymax = ymax * ratios_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


class GeneralizedRCNNTransform(nn.Module):

    def __init__(self, min_size, max_size, image_mean, image_std):
        super(GeneralizedRCNNTransform, self).__init__()
        if not isinstance(min_size, (tuple, list)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std

    def normalize(self, image):

        dtype, device = image.device, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)

        return (image - mean[:, None, None]) / std[:, None, None]

    def torch_choice(self, k):

        index = int(torch.empty(1).uniform_(0., float(len(k))).item())
        return k[index]

    def resize(self, image, target):

        h, w = image.shape[-2:]

        if self.training:
            size = float(self.torch_choice(self.min_size))
        else:
            size = float(self.min_size[-1])

        if torchvision._is_tracing():
            image = _resize_image_onnx(image, size, float(self.max_size))
        else:
            image = _resize_image(image, size, float(self.max_size))

        if target is None:
            return image, target

        bbox = target['boxes']
        bbox = resize_boxes(bbox, [h, w], image.shape[-2:])
        target['boxes'] = bbox

        return image, target

    @torch.jit.unused
    def _onnx_batch_images(self, images, size_divisible=32):
        pass

    def max_by_axis(self, the_list):
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
            return maxes

    def batch_images(self, images, size_divisible=32):

        if torchvision._is_tracing():
            return self._onnx_batch_images(images, size_divisible)

        max_size = self.max_by_axis([list(img.shape) for img in images])
        stride = float(size_divisible)

        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        batch_shape = [len(images)] + max_size

        batched_imgs = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)

        return batched_imgs

    def postprocess(self, result, image_shapes, original_image_sizes):

        if self.training:
            return result

        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred['boxes']
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]['boxes'] = boxes

        return result

    def __repr__(self):

        format_string = self.__class__.__name__ + '('
        _indent = '\n'
        format_string += '{}Nornalize(mean={},std={})'.format(
            _indent, self.image_mean, self.image_std)
        format_string += '{}Resize(min_size={},max_size={},mode="bilinear")'.format(
            _indent, self.min_size, self.max_size)
        format_string += '\n)'

        return format_string

    def forward(self, images, targets=None):

        images = [img for img in images]
        for i in range(len(images)):
            image = images[i]
            target_index = targets[i] if targets is not None else None

            if image.dim() != 3:
                raise ValueError('images is expected to be a list of 3d tensors')
            image = self.normalize(image)
            image, target_index = self.resize(image, target_index)
            images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index

        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images)
        image_sizes_list = torch.jt.annotate(List[Tuple[int, int]], [])

        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(images, image_sizes_list)

        return image_list, targets
