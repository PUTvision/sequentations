import albumentations as A

from sequentations.core.keypoints_utils import KeypointsProcessor

class Compose(A.Compose):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if self.processors.get("keypoints", None) is not None:
            self.processors["keypoints"] = KeypointsProcessor(self.processors["keypoints"].params)
