import torch
import torch.nn as nn


class DummyLoss(nn.Module):
    """Placeholder loss module for configuration compatibility."""

    def __init__(self):
        super().__init__()


class CELoss(nn.Module):
    """Regression loss with optional MoE auxiliary balancing term."""

    def __init__(self, kd_weight, koff_weight):
        super().__init__()
        self.kd_weight = kd_weight
        self.koff_weight = koff_weight

        self.MSEloss = torch.nn.MSELoss(reduction="mean")
        self.MAEloss = torch.nn.L1Loss(reduction="mean")
        self.HBloss = torch.nn.HuberLoss(reduction="mean")

    def forward(self, train_type, kd_pre, koff_pre, kd, koff, aux_dict, optimizer_idx, split="train"):
        """
        Compute training loss.

        Args:
            train_type (str): training mode string (e.g., includes "moe").
            kd_pre (Tensor): predicted affinity value.
            koff_pre (Tensor): predicted koff value (unused in current objective).
            kd (Tensor): target affinity value.
            koff (Tensor): target koff value (unused in current objective).
            aux_dict (Dict[str, Tensor]): auxiliary statistics from the model.
            optimizer_idx (int): optimizer index in multi-optimizer training.
            split (str): metric logging prefix.

        Returns:
            Tuple[Tensor, Dict[str, Tensor]]: scalar loss and logging dictionary.
        """
        if optimizer_idx == 0:
            if "moe" in train_type:
                coeff_balance = 0.05
                aux_loss = coeff_balance * aux_dict["importance_loss"]

                loss_kd = self.MSEloss(kd_pre, kd)
                loss = loss_kd * self.kd_weight + aux_loss

                log = {
                    f"{split}/loss": loss.clone().detach(),
                    f"{split}/kd_loss": loss_kd.clone().detach(),
                    f"{split}/aux_loss": aux_loss.clone().detach(),
                }

        return loss, log
