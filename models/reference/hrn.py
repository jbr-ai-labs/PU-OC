import torch
from torch import autograd

from base_models import OCModel
from classifier import Net

# code is adapted from https://github.com/morning-dews/HRN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calc_gradient_penalty(netD, real_data, fake_data, deg):
    BATCH_SIZE = real_data.shape[0]
    if real_data.dim() == 2:
        alpha = torch.rand(BATCH_SIZE, 1)
    else:
        alpha = torch.rand(BATCH_SIZE, 1, 1, 1)

    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(
                                  disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1)) ** deg).mean()

    return gradient_penalty


class HRN(OCModel):
    def __init__(self, model=Net, deg=12, lamda=0.5):
        super(HRN, self).__init__(model)
        self.deg = deg
        self.lamda = lamda

    def norm(self, batch):
        norm_batch = batch / torch.norm(batch, dim=0)
        return norm_batch

    def batch_loss(self, batch):
        batch_x = self.norm(batch[0].to(device))
        loss_pen = calc_gradient_penalty(self, batch_x, batch_x, self.deg)

        score_temp_0 = self(batch_x)

        mainloss_p = torch.log(score_temp_0 + 1e-8).mean()

        loss = - 1.0 * mainloss_p + self.lamda * loss_pen

        return loss
