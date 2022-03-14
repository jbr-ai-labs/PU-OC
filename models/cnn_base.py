import numpy as np
import torch
import torch.nn.functional as F

from models.base_models import OCModel, PUModelRandomBatch
from models.classifiers import Net_CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OC_CNN(OCModel):
    def __init__(self, model=Net_CNN, sigma=0.1):
        super().__init__(model, 0)
        self.sigma = sigma

    def _decision_function(self, batch_x):
        h = self.model.forward_start(batch_x.to(device))
        h = self.model.forward_end(h)
        return h

    def batch_loss(self, batch):
        batch_x = batch[0].to(device)
        out_pos = self.model.forward_start(batch_x)

        gaussian_data = np.random.normal(0, self.sigma, (int(batch_x.shape[0]), self.model.D))
        gaussian_data = torch.from_numpy(gaussian_data)
        out_neg = torch.autograd.Variable(gaussian_data.to(device)).float()

        out = torch.cat((out_pos, out_neg), 0)
        out = self.model.forward_end(out)

        labels = np.concatenate((np.ones((int(batch_x.shape[0]),)),
                                 np.zeros((int(batch_x.shape[0]),))),
                                axis=0)
        labels = torch.from_numpy(labels)
        labels = torch.autograd.Variable(labels.to(device))
        labels = labels.unsqueeze(1).to(device)

        loss = F.binary_cross_entropy(out, labels.float())

        return loss


class PU_CNN(PUModelRandomBatch):
    def __init__(self, model=Net_CNN, alpha=0.5):
        self.alpha = alpha
        super().__init__(model, 0)

    def _decision_function(self, batch_x):
        h = self.model.forward_start(batch_x.to(device))
        h = self.model.forward_end(h)
        return h

    def batch_loss(self, batch):
        batch_x = batch[0].to(device)
        batch_s = batch[2].to(device)

        out = self.model.forward_start(batch_x)
        out = self.model.forward_end(out)

        pred_p = out[batch_s == 1]
        pred_u = out[batch_s == self.neg_label]

        target_pp = torch.ones_like(pred_p)
        target_pu = torch.zeros_like(pred_p)
        target_uu = torch.zeros_like(pred_u)

        loss_p = self.alpha * F.binary_cross_entropy(pred_p, target_pp)
        loss_n = F.binary_cross_entropy(pred_u, target_uu) - self.alpha * F.binary_cross_entropy(pred_p, target_pu)

        if loss_n > 0:
            return loss_p + loss_n
        return loss_p
