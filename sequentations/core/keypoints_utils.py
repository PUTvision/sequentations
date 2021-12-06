from albumentations.core.utils import DataProcessor as ADataProcessor
import albumentations.augmentations.keypoints_utils as U

class DataProcessor(ADataProcessor):
    def preprocess(self, data):
        data = self.add_label_fields_to_data(data)

        rows, cols = data["image"].shape[1:3]
        for data_name in self.data_fields:
            data[data_name] = self.check_and_convert(data[data_name], rows, cols, direction="to")


class KeypointsProcessor(DataProcessor):
    @property
    def default_data_name(self):
        return "keypoints"

    def ensure_data_valid(self, data):
        if self.params.label_fields:
            if not all(i in data.keys() for i in self.params.label_fields):
                raise ValueError(
                    "Your 'label_fields' are not valid - them must have same names as params in "
                    "'keypoint_params' dict"
                )

    def ensure_transforms_valid(self, transforms):
        # IAA-based augmentations supports only transformation of xy keypoints.
        # If your keypoints formats is other than 'xy' we emit warning to let user
        # be aware that angle and size will not be modified.

        try:
            from albumentations.imgaug.transforms import DualIAATransform
        except ImportError:
            # imgaug is not installed so we skip imgaug checks.
            return

        if self.params.format is not None and self.params.format != "xy":
            for transform in transforms:
                if isinstance(transform, DualIAATransform):
                    break

    def filter(self, data, rows, cols):
        return U.filter_keypoints(data, rows, cols, remove_invisible=self.params.remove_invisible)

    def check(self, data, rows, cols):
        return U.check_keypoints(data, rows, cols)

    def convert_from_albumentations(self, data, rows, cols):
        return U.convert_keypoints_from_albumentations(
            data,
            self.params.format,
            rows,
            cols,
            check_validity=self.params.remove_invisible,
            angle_in_degrees=self.params.angle_in_degrees,
        )

    def convert_to_albumentations(self, data, rows, cols):
        return U.convert_keypoints_to_albumentations(
            data,
            self.params.format,
            rows,
            cols,
            check_validity=self.params.remove_invisible,
            angle_in_degrees=self.params.angle_in_degrees,
        )
