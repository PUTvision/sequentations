import numpy as np
import cv2

import sys
sys.path.insert(0, '..')

from sequentations.augmentations.transforms import Flip, Rotate, ColorJitter, RandomGamma
from sequentations.core.composition import Sequential


def main():
    cap = cv2.VideoCapture('./data/180607_A_101.mp4')

    # SEQUENTATIONS
    aug = Sequential([
        Rotate(limit=10, always_apply=True, border_mode=cv2.BORDER_CONSTANT),
        Flip(always_apply=True),
        ColorJitter(brightness=0, contrast=0, hue=0.01, saturation=0.5, always_apply=True),
        RandomGamma(gamma_limit=(80, 120), always_apply=True)
    ])

    if not cap.isOpened():
        print("Error opening video stream or file")
        exit(1)

    sequence = []

    while(cap.isOpened()):
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        sequence.append(frame)

    # When everything done, release the video capture object
    cap.release()

    sequence = np.array(sequence, dtype=np.uint8)

    transformed = aug(image=sequence)
    sequence = transformed['image']

    for seq in sequence:
        cv2.imshow('image', frame)

        cv2.waitKey(24)

    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
