# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target=None):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        if target is None:
            return image
        target = target.resize(image.size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.vflip(image)
            target = target.transpose(1)
        return image, target


class RandomCrop(object):
    def __init__(self, size, pad_if_needed=False, fill=0, padding_mode='constant'):
        """

        :param size: int or sequence (width, height)
        :param pad_if_needed: bool
        :param fill: integer
        :param padding_mode: string
        """
        self.size = size if type(size) == int else (size, size)
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, image, target):
        width, height = image.size
        x_start = random.randint(0, width - self.size[0] - 1)
        y_start = random.randint(0, height - self.size[1] - 1)

        image = image.crop(x_start, y_start, x_start + self.size[0], y_start + self.size[1])

        # Remove empty instance masks
        target = target[:, y_start: y_start + self.size[1], x_start: x_start + self.size[0]]

        valid_rows = target.sum(axis=-1).sum(axis=-1) > 0




class Pad(object):
    def __init__(self, size):

        pass


class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue, )

    def __call__(self, image, target):
        image = self.color_jitter(image)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image
        return image, target


def func():
    import numpy as np
    a = np.array([[[1, 1],
                   [0, 0],
                   [1, 0]],
                  [[0, 0],
                   [1, 1],
                   [1, 0]],
                  [[0, 0],
                   [0, 0],
                   [0, 0]],
                  [[1, 0],
                   [1, 1],
                   [1, 0]]])
    print(a, a.shape)
    summed = a.sum(axis=-1).sum(axis=-1)
    print(summed)
    print(summed > 0)

    print(a[summed > 0, :, :])


if __name__ == '__main__':
    func()
