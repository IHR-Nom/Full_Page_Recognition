import torch
from typing import Optional, List

from PIL import Image
from torch import Tensor
import torchvision as tv
import cv2
import json
import os
import numpy as np
MAX_DIM = 299


def read_json(file_name):
    with open(file_name) as handle:
        out = json.load(handle)
    return out


def nested_tensor_from_tensor_list(tensor_list: List[Tensor], img_width, img_height):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = [1, 299, 343]
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


val_transform = tv.transforms.Compose([
    tv.transforms.Resize(299),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def padding_image(image_path, config):
    with Image.open(os.path.join(image_path)) as img:
        img = img.convert('RGB')
        width, height = img.size
        # Resize image
        n_width, n_height = int(width / 2), int(height / 2)
        resized_img = img.resize((n_width, n_height), Image.ANTIALIAS)
        padding_img = Image.new('RGB', (config.max_img_w, config.max_img_h), (0, 0, 0))
        padding_img.paste(resized_img)
        return padding_img


def padding_image_v2(img, expected_size):
    img = img.convert('L')
    original_w, original_h = img.size
    expected_w, expected_h = expected_size
    ratio_w, ratio_h = expected_w / original_w, expected_h / original_h
    if ratio_w < ratio_h:
        new_w, new_h = expected_w, original_h * ratio_w
    else:
        new_w, new_h = original_w * ratio_h, expected_h
    img = img.resize((int(new_w), int(new_h)), Image.ANTIALIAS)
    padding_img = Image.new('RGB', expected_size, (0, 0, 0))
    padding_img.paste(img)
    return padding_img


def resize_filling(image, new_size, color=None):
    n_width, n_height = new_size
    height, width = image.shape[:2]
    ratio_w = n_width / width
    ratio_h = n_height / height
    ratio = min(ratio_h, ratio_w)
    image = cv2.resize(image, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    height, width = image.shape[:2]
    blank_image = np.zeros((n_height, n_width, 3), np.uint8)
    if color is None:
        color = bincount_app(image)
    lower = np.array([color[0] - 20, color[1] - 20, color[2] - 20])
    upper = np.array([color[0] + 20, color[1] + 20, color[2] + 20])
    mask = cv2.inRange(image, lower, upper)
    masked_image = np.copy(image)
    masked_image[mask != 0] = color

    # img_bw = 255 * (cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY) > 10).astype('uint8')
    #
    # se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # mask = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, se1)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
    #
    # mask = np.dstack([mask, mask, mask]) / 255
    # out = masked_image * mask
    #
    blank_image[:] = color

    x_offset, y_offset = int((n_width - width) / 2), int((n_height - height) / 2)
    # Here, y_offset+height <= blank_image.shape[0] and x_offset+width <= blank_image.shape[1]
    blank_image[y_offset:y_offset + height, x_offset:x_offset + width] = masked_image.copy()
    # plt.figure()
    # plt.imshow(blank_image)
    #
    # plt.axis('off')
    # plt.ioff()
    # # plt.pause(0.05)
    # # plt.clf()
    # plt.show()
    return blank_image


def bincount_app(a):
    image_to_array = np.array(a)
    a2D = image_to_array.reshape(-1, image_to_array.shape[-1])
    col_range = (256, 256, 256) # generically : a2D.max(0)+1
    a1D = np.ravel_multi_index(a2D.T, col_range)
    return np.unravel_index(np.bincount(a1D).argmax(), col_range)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)