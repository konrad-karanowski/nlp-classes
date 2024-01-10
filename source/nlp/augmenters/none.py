import torch



class NoneAugmenter:

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x
