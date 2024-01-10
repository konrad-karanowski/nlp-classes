import torch
import numpy as np

from nlp.lightning_model.flow_augmentation_model import FlowAugmentationModel



class FlowAugmenter:

    def __init__(
        self,
        checkpoint_path: str,
        num_classes: int,
        alpha: float,
        perc_time: float,
        *args,
        **kwargs
    ) -> None:
        
        self.flow = FlowAugmentationModel.load_from_checkpoint(checkpoint_path).flow
        self.flow.eval()

        self.num_classes = num_classes
        self.alpha = alpha
        self.perc_time = perc_time

    def __call__(self, x: torch.Tensor, y: torch.tensor) -> torch.Tensor:
        if np.random.rand() <= self.perc_time:
            with torch.no_grad():
                y = torch.nn.functional.one_hot(y, self.num_classes).float()
                z = self.flow.transform_to_noise(x, context=y)

                z_hat = z + self.alpha * torch.randn_like(z)

                x_hat, _ = self.flow._transform.inverse(z_hat, context=y)
            return x_hat
        return x

