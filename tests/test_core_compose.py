import pytest
import numpy as np
import torch
import cv2

from sequentations.core.composition import Compose, Sequential
from sequentations.pytorch.transforms import ToTensorV2
from sequentations.augmentations.transforms import ColorJitter, Flip, Normalize, RandomGamma, Rotate


def test_compose_normalize():
    arr = np.ones((1, 100, 100, 3), dtype=np.uint8) * 255
    aug = Compose([
        Normalize(mean=(0, 0, 0), std=(1, 1, 1))
    ])
    data = aug(image=arr, force_apply=True)['image']

    assert np.isclose(data.max(), 1.0)
    assert np.isclose(data.mean(), 1.0)
    assert np.isclose(data.std(), 0.0)
    assert np.isclose(data.min(), 1.0)

    assert data.dtype == np.float32


def test_sequential_normalize():
    arr = np.ones((1, 100, 100, 3), dtype=np.uint8) * 255
    aug = Sequential([
        Normalize(mean=(0, 0, 0), std=(1, 1, 1))
    ])
    data = aug(image=arr, force_apply=True)['image']

    assert np.isclose(data.max(), 1.0)
    assert np.isclose(data.mean(), 1.0)
    assert np.isclose(data.std(), 0.0)
    assert np.isclose(data.min(), 1.0)

    assert data.dtype == np.float32


def test_compose_torch_to_tensor_v2_augmentations_with_transpose_3d_mask():

    aug = Compose([
        ToTensorV2(transpose_mask=True)
    ])
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


def test_sequential_torch_to_tensor_v2_augmentations_with_transpose_3d_mask():

    aug = Sequential([
        ToTensorV2(transpose_mask=True)
    ])
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


def test_compose_many():
    _augmentations = Compose([
        Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT),
        RandomGamma(gamma_limit=(80, 120)),
        ColorJitter(brightness=0, contrast=0, hue=0.01, saturation=0.5),
        Flip(),
    ])

    _transforms = Compose([
        Normalize([0.4477, 0.4209, 0.3906], [0.2767, 0.2695, 0.2714]),
        ToTensorV2(transpose_mask=True)
    ])

    frame = np.array(np.random.rand(4, 448, 800, 3) * 255, dtype=np.uint8)
    label = np.array(np.random.rand(1, 448, 800, 1) * 255, dtype=np.uint8)

    result = _augmentations(image=frame, mask=label)
    frame, label = result['image'], result['mask']

    transformed = _transforms(image=frame, mask=label)
    frame, label = transformed['image'], transformed['mask']


def test_sequential_many():
    _augmentations = Sequential([
        Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT),
        RandomGamma(gamma_limit=(80, 120)),
        ColorJitter(brightness=0, contrast=0, hue=0.01, saturation=0.5),
        Flip(),
    ])

    _transforms = Sequential([
        Normalize([0.4477, 0.4209, 0.3906], [0.2767, 0.2695, 0.2714]),
        ToTensorV2(transpose_mask=True)
    ])

    frame = np.array(np.random.rand(4, 448, 800, 3) * 255, dtype=np.uint8)
    label = np.array(np.random.rand(1, 448, 800, 1) * 255, dtype=np.uint8)

    result = _augmentations(image=frame, mask=label)
    frame, label = result['image'], result['mask']

    transformed = _transforms(image=frame, mask=label)
    frame, label = transformed['image'], transformed['mask']
