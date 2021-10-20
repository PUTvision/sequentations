import torch
import albumentations as A


class ToTensorV2(A.DualTransform):
    def __init__(self, transpose_mask=False, always_apply=True, p=1):
        super().__init__(always_apply, p)
        self.transpose_mask = transpose_mask

    @property
    def targets(self):
        return {'image': self.apply, 'mask': self.apply_to_mask}

    def apply(self, img, **params):
        return torch.from_numpy(img.transpose(3, 0, 1, 2))

    def apply_to_mask(self, mask, **params):
        if self.transpose_mask:
            mask = mask.transpose(3, 0, 1, 2)

        return torch.from_numpy(mask)
