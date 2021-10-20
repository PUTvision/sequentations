from typing import Type

import numpy as np
import albumentations as A


def SequentialWrapper(cls):
    class DualTransform(cls):
        def __init__(self, always_apply=False, p=0.5, *args, **kwargs):
            super().__init__(always_apply=always_apply, p=p, *args, **kwargs)
            self.transform = super()

        def apply(self, img, *args, **kwargs):
            if len(img.shape) != 4:
                return img

            kwargs['cols'], kwargs['rows'] = img.shape[1:3]
            return np.stack(tuple(map(lambda i: super(cls, self).apply(i, *args, **kwargs), img)))

        def apply_to_mask(self, mask, *args, **kwargs):
            if len(mask.shape) != 4:
                return mask

            kwargs['cols'], kwargs['rows'] = mask.shape[1:3]
            return np.stack(tuple(map(lambda m: super(cls, self).apply_to_mask(m, *args, **kwargs), mask)))

    return DualTransform
