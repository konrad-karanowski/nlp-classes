from typing import Any, Dict, Tuple, Union, Sequence, Optional
import os

import hydra
import torch
import lightning as pl
from torch.optim import Optimizer
from omegaconf import DictConfig
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


class FlowAugmentationModel(pl.LightningModule):


    def __init__(
        self,
        compile: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:
        super(FlowAugmentationModel, self).__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.flow = hydra.utils.instantiate(self.hparams.flow)

    def configure_optimizers(
            self,
        ) -> Union[Sequence[Optimizer], Tuple[Sequence[Optimizer], Sequence[Any]]]:
            params = self.flow.parameters()
            optimizer = hydra.utils.instantiate(
                self.hparams.optimizer, params=params, _convert_="partial"
            )
            # return only optimizer if lr_scheduler is not provided.
            if "lr_scheduler" not in self.hparams:
                return {'optimizer': optimizer}
            scheduler = hydra.utils.instantiate(
                self.hparams.lr_scheduler, optimizer=optimizer, _convert_="partial"
            )
            # reduce LR on Plateau requires special treatment
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                return {
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                    "monitor": self.hparams.monitor.metric,
                }
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.flow.log_prob(inputs=x, context=y)
    
    # def sample(self, x: torch.Tensor, y: torch.Tensor, n: int, bs: int) -> torch.Tensor:
    #     return self.flow.sample(batch=x, y=y, num_samples=n, batch_size=bs)

    def _shared_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch['x'], batch['y']
        loss = - self.flow.log_prob(inputs=x.float(), context=y).mean()
        return loss


    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss = self._shared_step(batch)

        # update and log metrics
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss = self._shared_step(batch)

        # update and log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss = self._shared_step(batch)

        # update and log metrics
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss


@hydra.main(config_path=os.environ["CONFIG_DIR"], config_name="default")
def _run(config: DictConfig) -> None:
    """
    Run to test if the module works.
    """
    flow = hydra.utils.instantiate(config.lightning_model, _recursive_=False)
    print(f'{flow}')


if __name__ == "__main__":
    _run()
