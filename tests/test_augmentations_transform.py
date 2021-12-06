import pytest
import numpy as np

import context
from sequentations.augmentations.transforms import RandomGamma, ColorJitter, Normalize, RandomResizedCrop, Resize


def test_random_gamma():
    arr = np.full((2, 100, 100, 3), fill_value=127, dtype=np.uint8)
    aug = RandomGamma()
    data = aug(image=arr, force_apply=True)['image']

    assert np.shape(data) == np.shape(arr)
    assert np.count_nonzero(data != 127) != 0

    first_slice = arr[0]
    for current_slice in arr[1:]:
        assert np.all(first_slice == current_slice)


def test_color_jitter():
    arr = np.full((2, 100, 100, 3), fill_value=127, dtype=np.uint8)
    aug = ColorJitter()
    data = aug(image=arr, force_apply=True)['image']

    assert np.shape(data) == np.shape(arr)
    assert np.count_nonzero(data != 127) != 0

    first_slice = arr[0]
    for current_slice in arr[1:]:
        assert np.all(first_slice == current_slice)


def test_normalize_v2():
    arr = np.zeros((2, 100, 100, 3), dtype=np.uint8)
    aug = Normalize(mean=(0, 0, 0), std=(1, 1, 1))
    data = aug(image=arr, force_apply=True)['image']

    assert np.isclose(data.max(), 0.0)
    assert np.isclose(data.mean(), 0.0)
    assert np.isclose(data.std(), 0.0)
    assert np.isclose(data.min(), 0.0)

    arr = np.ones((1, 100, 100, 3), dtype=np.uint8) * 255
    aug = Normalize(mean=(0, 0, 0), std=(1, 1, 1))
    data = aug(image=arr, force_apply=True)['image']

    assert np.isclose(data.max(), 1.0)
    assert np.isclose(data.mean(), 1.0)
    assert np.isclose(data.std(), 0.0)
    assert np.isclose(data.min(), 1.0)


def test_random_resized_crop():
    arr = np.full((2, 1080, 1920, 3), fill_value=127, dtype=np.uint8)
    mask = np.full((2, 1080, 1920, 1), fill_value=0, dtype=np.uint8)
    mask[0, 10, 10] = 100
    aug = RandomResizedCrop(height=640, width=640)
    data = aug(image=arr, mask=mask, force_apply=True)
    

    assert np.shape(data['image']) == (2, 640, 640, 3)
    assert np.shape(data['mask']) == (2, 640, 640, 1)

    assert np.sum(data['mask'][0]) == 100


def test_resize():
    arr = np.full((2, 1080, 1920, 3), fill_value=127, dtype=np.uint8)
    mask = np.full((2, 1080, 1920, 1), fill_value=0, dtype=np.uint8)
    mask[0, 10, 10] = 100

    aug = Resize(height=640, width=640)
    data = aug(image=arr, mask=mask, force_apply=True)

    assert np.shape(data['image']) == (2, 640, 640, 3)
    assert np.shape(data['mask']) == (2, 640, 640, 1)

    assert np.sum(data['mask'][0]) == 100
