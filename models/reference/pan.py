import torch
import torch.optim as opt

from models.base_models import PUModelRandomBatch
from classifier import Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PAN(PUModelRandomBatch):
    def __init__(self, model=Net, clf=Net, lamda=1e-3):
        """
        :param model: classifier which is treated as discriminator
        """
        super(PAN, self).__init__(model)
        self.clf = clf()
        self.lamda = lamda
        self.eps = 1e-8

    def batch_loss(self, batch):
        batch_x, _, batch_s = batch
        batch_x = batch_x.to(device).float()

        batch_p = batch_x[batch_s == 1]
        batch_u = batch_x[batch_s == self.neg_label]

        pred_d_p = self.model(batch_p)
        pred_d_u = self.model(batch_u)

        pred_c_u = self.clf(batch_u)

        term1 = torch.log(pred_d_p + self.eps).sum() + torch.log(1 - pred_d_u + self.eps).sum()

        term2_d = ((1 - 2 * pred_d_u) * torch.log(pred_c_u.detach() + self.eps)).sum() + \
                  ((2 * pred_d_u - 1) * torch.log(1 - pred_c_u.detach() + self.eps)).sum()

        term2_c = ((1 - 2 * pred_d_u.detach()) * torch.log(pred_c_u + self.eps)).sum() + \
                  ((2 * pred_d_u.detach() - 1) * torch.log(1 - pred_c_u + self.eps)).sum()

        return -term1 - self.lamda * term2_d + self.lamda * term2_c - (self.lamda * term2_c).detach()

    def get_optimizers(self, lr, **kwargs):
        self.optimizers = [opt.Adam(self.model.parameters(), lr=kwargs['lr_d']),
                           opt.Adam(self.clf.parameters(), lr=kwargs['lr_c'])]

    def get_schedulers(self, gamma, **kwargs):
        self.schedulers = [opt.lr_scheduler.ExponentialLR(self.optimizers[0], gamma=kwargs['gamma_d']),
                           opt.lr_scheduler.ExponentialLR(self.optimizers[1], gamma=kwargs['gamma_c'])]

    def fit(self,
            train_data,
            num_epochs=20,
            lr=1e-3,
            lr_c=None,
            lr_d=None,
            batch_size=512,
            gamma=0.99,
            verbose=False,
            test_data=None,
            gamma_d=None,
            gamma_c=None,
            **kwargs):

        if lr_c is None or lr_d is None:
            lr_c = lr
            lr_d = lr
        if gamma_c is None or gamma_d is None:
            gamma_c = gamma
            gamma_d = gamma
        super().fit(train_data,
                    num_epochs=num_epochs,
                    lr_c=lr_c,
                    lr_d=lr_d,
                    batch_size=batch_size,
                    gamma=gamma,
                    verbose=verbose,
                    test_data=test_data,
                    gamma_d=gamma_d,
                    gamma_c=gamma_c,
                    **kwargs)
