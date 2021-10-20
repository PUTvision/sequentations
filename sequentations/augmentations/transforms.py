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
