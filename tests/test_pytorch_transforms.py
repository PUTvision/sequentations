import pytest
import numpy as np
import torch

import context
from sequentations.pytorch.transforms import ToTensorV2


def test_torch_to_tensor_v2_augmentations_with_transpose_3d_mask():

    aug = ToTensorV2(transpose_mask=True)
    image = np.random.randint(low=0, high=256, size=(4, 100, 100, 3), dtype=np.uint8)
    mask = np.random.randint(low=0, high=256, size=(4, 100, 100, 1), dtype=np.uint8)

    data = aug(image=image, mask=mask, force_apply=True)

    image_batch, image_height, image_width, image_num_channels = image.shape
    mask_batch, mask_height, mask_width, mask_num_channels = mask.shape

    assert isinstance(data["image"], torch.Tensor) and data["image"].shape == (
        image_num_channels,
        image_batch,
        image_height,
        image_width,
    )
    assert isinstance(data["mask"], torch.Tensor) and data["mask"].shape == (
        mask_num_channels,
        mask_batch,
        mask_height,
        mask_width,
    )
    assert data["image"].dtype == torch.uint8
    assert data["mask"].dtype == torch.uint8
