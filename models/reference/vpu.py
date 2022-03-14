import numpy as np
import torch
import torch.optim as opt
from sklearn import metrics
from torch import nn

from classifier import Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VPU(nn.Module):
    def __init__(self, model=Net, alpha=0.5, lam=0.03):
        super().__init__()
        self.alpha = alpha
        self.lam = lam
        self.model = model().to(device)

    def decision_function_loader(self, test_loader):
        with torch.no_grad():
            for idx, (data, target, _) in enumerate(test_loader):
                data = data.to(device).float()
                log_phi = self.model(data)
                if idx == 0:
                    log_phi_all = log_phi
                    target_all = target
                else:
                    log_phi_all = torch.cat((log_phi_all, log_phi))
                    target_all = torch.cat((target_all, target))

        return log_phi_all.cpu().detach().numpy(), target_all.cpu().detach().numpy()

    def decision_function(self, data):
        loader = torch.utils.data.DataLoader(data,
                                             batch_size=512)

        return self.decision_function_loader(loader)

    def fit(self,
            train_data,
            num_epochs=50,
            lr=5e-3,
            gamma=0.99,
            test_data=None,
            batch_size=512,
            verbose=False,
            **kwargs):

        optim = opt.Adam(self.model.parameters(), lr=lr)
        scheduler = opt.lr_scheduler.ExponentialLR(optim, gamma=gamma)

        lab_data = train_data.lab_data(lab=1)
        unl_data = train_data.lab_data(lab=0)

        p_loader = torch.utils.data.DataLoader(lab_data,
                                               batch_size=batch_size,
                                               shuffle=True)

        u_loader = torch.utils.data.DataLoader(unl_data,
                                               batch_size=batch_size,
                                               shuffle=True)

        test_loader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=batch_size,
                                                  shuffle=False)
        for epoch in range(num_epochs):
            self.model.train()
            tl = 0
            vl = 0
            rl = 0

            for batch in zip(p_loader, u_loader):
                data_p, data_x = batch[0][0].to(device), batch[1][0].to(device)
                if len(data_x) != len(data_p):
                    continue

                pred_p = self.model(data_p)
                pred_u = self.model(data_x)

                l_var = torch.log(pred_u.mean()) - torch.log(pred_p).mean()

                gamma = np.random.beta(self.alpha, self.alpha)
                gamma = 1 - gamma

                ind_shuffle = torch.randperm(data_p.shape[0])

                tilde_x = gamma * data_p[ind_shuffle] + (1 - gamma) * data_x
                tilde_pred = gamma + (1 - gamma) * pred_u

                l_reg = ((torch.log(self.model(tilde_x)) - torch.log(tilde_pred)) ** 2).mean()

                l_tot = l_var + self.lam * l_reg

                optim.zero_grad()
                l_tot.backward()
                optim.step()

                tl += l_tot.item()
                vl += l_var.item()
                rl += l_reg.item()

            if verbose:
                y_pred, y_test = self.decision_function_loader(test_loader)
                auc = metrics.roc_auc_score(y_test, y_pred)
                print(f"[{epoch}/{num_epochs}]: total_loss={tl:.4f} var_loss={vl:.4f} reg_loss={rl:.4f} auc={auc:.4f}")
            scheduler.step()
