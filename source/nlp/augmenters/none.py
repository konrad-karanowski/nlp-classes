import torch



class NoneAugmenter:

    def __init__(self, *args, **kwargs) -> None:
        pass

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        return x
