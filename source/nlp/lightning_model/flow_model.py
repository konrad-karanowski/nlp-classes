from typing import Any, Dict, Tuple, Union, Sequence, Optional
import os

import hydra
import torch
import lightning as pl
from torch.optim import Optimizer
from omegaconf import DictConfig
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


class FlowModel(pl.LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        compile: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:
        """Initialize a `MNISTLitModule`.

        Args:
            compile (Optional[bool], optional). Whether to compile the flow. Defaults to False.
            *args, **kwargs: Hyperaparameters for the flow. Saved by `self.save_hyperparameters()`
        """
        super(FlowModel, self).__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.flow = hydra.utils.instantiate(self.hparams.flow)

    def configure_optimizers(
            self,
        ) -> Union[Sequence[Optimizer], Tuple[Sequence[Optimizer], Sequence[Any]]]:
            """Configure optimizer and lr scheduler.
            Returns:
                Union[Sequence[Optimizer], Tuple[Sequence[Optimizer], Sequence[Any]]]:
                    Optimizer or optimizer and lr scheduler.
            """
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
        """Performs a forward pass.

        Args:
            x (torch.Tensor): Input.

        Returns:
            torch.Tensor: Output
        """
        return self.flow.log_prob(inputs=x, context=y)
    
    # def sample(self, x: torch.Tensor, y: torch.Tensor, n: int, bs: int) -> torch.Tensor:
    #     return self.flow.sample(batch=x, y=y, num_samples=n, batch_size=bs)

    def _shared_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single flow step on a batch of data.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A batch of data (a tuple) containing the input tensor of images and target labels.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch['x'], batch['y']
        y = torch.nn.functional.one_hot(y, self.hparams.num_classes).float()
        loss = - self.flow.log_prob(inputs=y, context=x).mean()
        return loss


    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A batch of data (a tuple) containing the input tensor of images and target
                labels.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: A tensor of losses between flow predictions and targets.
        """
        loss = self._shared_step(batch)

        # update and log metrics
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the training set.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A batch of data (a tuple) containing the input tensor of images and target
                labels.
            batch_idx (int): The index of the current batch.
        """
        loss = self._shared_step(batch)

        # update and log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the training set.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A batch of data (a tuple) containing the input tensor of images and target
                labels.
            batch_idx (int): The index of the current batch.
        """
        loss = self._shared_step(batch)

        # update and log metrics
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        Args:
            stage (str): Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.flow = torch.compile(self.flow)



@hydra.main(config_path=os.environ["CONFIG_DIR"], config_name="default")
def _run(config: DictConfig) -> None:
    """
    Run to test if the module works.
    """
    flow = hydra.utils.instantiate(config.lightning_model, _recursive_=False)
    print(f'{flow}')


if __name__ == "__main__":
    _run()
