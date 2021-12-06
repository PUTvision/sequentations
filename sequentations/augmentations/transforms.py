import numpy as np
import cv2

import albumentations as A
import albumentations.augmentations as aug

from ..core.transforms_interface import SequentialWrapper


@SequentialWrapper
class ColorJitter(aug.ColorJitter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@SequentialWrapper
class Flip(aug.Flip):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@SequentialWrapper
class RandomGamma(aug.RandomGamma):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RandomResizedCrop(aug.RandomResizedCrop):
    def __init__(self,
                 height,
                 width,
                 scale=(0.08, 1.0), 
                 ratio=(0.75, 1.3333333333333333),
                 interpolation=1,
                 always_apply=False,
                 p=1
                 ):
        super().__init__(always_apply, p)
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    def apply(self, img, crop_height=0, crop_width=0, h_start=0, w_start=0, interpolation=1, **params):
        return np.stack(tuple(map(lambda m: aug.geometric.functional.resize(aug.random_crop(m, crop_height, crop_width, h_start, w_start), height=self.height, width=self.width, interpolation=interpolation), img)))

    def apply_to_mask(self, mask, crop_height=0, crop_width=0, h_start=0, w_start=0,interpolation=1, **params):
        return np.stack(tuple(map(lambda m: aug.geometric.functional.resize(aug.random_crop(m, crop_height, crop_width, h_start, w_start), height=self.height, width=self.width, interpolation=cv2.INTER_NEAREST_EXACT), mask)))

    def get_transform_init_args_names(self):
        return ("height", "width", "scale", "ratio", "interpolation")


class Resize(aug.Resize):
    def __init__(self,
                 height,
                 width,
                 interpolation=1,
                 always_apply=False,
                 p=1
                 ):
        super().__init__(always_apply, p)
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def apply(self, img, interpolation=1, **params):
        return np.stack(tuple(map(lambda m: aug.geometric.functional.resize(m, height=self.height, width=self.width, interpolation=interpolation), img)))

    def apply_to_mask(self, mask, interpolation=1, **params):
        return np.stack(tuple(map(lambda m: aug.geometric.functional.resize(m, height=self.height, width=self.width, interpolation=cv2.INTER_NEAREST_EXACT), mask)))

    def get_transform_init_args_names(self):
        return ("height", "width", "interpolation")


@SequentialWrapper
class Rotate(aug.Rotate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Normalize(A.ImageOnlyTransform):
    def __init__(
        self,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        max_pixel_value=255.0,
        always_apply=False,
        p=1.0,
    ):
        super(Normalize, self).__init__(always_apply, p)
        self.mean = mean
        self.std = std
        self.max_pixel_value = max_pixel_value

    def apply(self, image, **params):
        return aug.functional.normalize(image, self.mean, self.std, self.max_pixel_value)

    def get_transform_init_args_names(self):
        return ("mean", "std", "max_pixel_value")
