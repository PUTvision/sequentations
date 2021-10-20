import albumentations as A



class Sequential(A.Sequential):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

class Compose(A.Compose):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
