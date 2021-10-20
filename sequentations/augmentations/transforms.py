import albumentations.augmentations as A

from ..core.transforms_interface import SequentialWrapper


@SequentialWrapper
class ColorJitter(A.ColorJitter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@SequentialWrapper
class Flip(A.Flip):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@SequentialWrapper
class RandomGamma(A.RandomGamma):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@SequentialWrapper
class Rotate(A.Rotate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
