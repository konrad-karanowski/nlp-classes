import torch

from nlp.lightning_model.flow_augmentation_model import FlowAugmentationModel



class FlowAugmenter:

    def __init__(
        self,
        checkpoint_path: str,
        *args,
        **kwargs
    ) -> None:
        self.flow = FlowAugmentationModel.load_from_checkpoint(checkpoint_path).flow

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        pass
