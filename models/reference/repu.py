from abc import ABC

import torch
import torch.nn.functional as F

from models.base_models import PUModelRandomBatch
from ..classifiers import Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class REPU(PUModelRandomBatch, ABC):
    def __init__(self, model=Net, alpha=0.5):
        super().__init__(model)
        self.alpha = alpha

    def _batch_loss(self, batch):
        batch_x, _, batch_s = batch
        batch_x = batch_x.to(device).float()

        batch_p = batch_x[batch_s == 1]
        batch_u = batch_x[batch_s == self.neg_label]

        pred_p = self(batch_p)
        pred_u = self(batch_u)

        target_pp = torch.ones_like(pred_p)
        target_pu = torch.zeros_like(pred_p)
        target_uu = torch.zeros_like(pred_u)

        loss_p = self.alpha * F.binary_cross_entropy(pred_p, target_pp)
        loss_n = F.binary_cross_entropy(pred_u, target_uu) - self.alpha * F.binary_cross_entropy(pred_p, target_pu)

        return loss_p, loss_n


class nnPU(REPU):
    def batch_loss(self, batch):
        loss_p, loss_n = self._batch_loss(batch)
        if loss_n < 0:
            return loss_p
        return loss_p + loss_n


class uPU(REPU):
    def batch_loss(self, batch):
        loss_p, loss_n = self._batch_loss(batch)
        return loss_p + loss_n
