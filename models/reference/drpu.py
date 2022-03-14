import torch
import torch.nn as nn
import torch.optim as opt
import torch.utils.data
from sklearn import metrics

# code is adapted from https://github.com/csnakajima/pu-learning
from classifier import Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BregmanDivergence(object):
    def __init__(self):
        # self.f = f_df[0]
        # self.df = f_df[1]
        self.f = lambda x: torch.square(x - 1) / 2
        self.df = lambda x: x - 1
        self.L = None

    def __call__(self, y_p, y_u):
        E_p = torch.mean(-self.df(y_p))
        E_u = torch.mean(y_u * self.df(y_u) - self.f(y_u))
        self.L = E_p + E_u
        return self.L

    def value(self):
        return self.L.item()


class NonNegativeBregmanDivergence(BregmanDivergence):
    def __init__(self, alpha, thresh=0, weight=1):
        super().__init__()
        self.alpha = alpha
        self.thresh = thresh
        self.weight = weight

        self.f_dual = lambda x: x * self.df(x) - self.f(x)
        self.f_nn = lambda x: self.f_dual(x) - self.f_dual(0 * x)

    def __call__(self, y_p, y_u):
        E_pp = torch.mean(-self.df(y_p) + self.alpha * self.f_nn(y_p))
        E_pn = torch.mean(self.f_nn(y_p))
        E_u = torch.mean(self.f_nn(y_u))
        self.L = E_pp + max(0, E_u - self.alpha * E_pn) + self.f_dual(0 * E_u)
        return self.L if E_u - self.alpha * E_pn >= self.thresh else self.weight * (self.alpha * E_pn - E_u)


class DRPU(nn.Module):
    def __init__(self, model=Net, alpha=0.001):
        super().__init__()
        self.model = model(prob=False)
        self.alpha = alpha
        self.criterion = NonNegativeBregmanDivergence(alpha)
        self.criterion_val = BregmanDivergence()

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
            num_epochs=60,
            lr=1e-3,
            test_data=None,
            batch_size=512,
            gamma=0.99,
            verbose=False,
            **kwargs):

        self.to(device)
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

            for batch in zip(p_loader, u_loader):

                batch_p = batch[0][0].to(device).float()
                batch_u = batch[1][0].to(device).float()

                if len(batch_p) != len(batch_u):
                    pass

                len_p, len_u = len(batch_p), len(batch_u)

                batch_x = torch.cat((batch_p, batch_u))

                pred = self.model(batch_x).view(-1)
                pred_p, pred_u = pred[: len_p], pred[len_p:]

                loss = self.criterion(pred_p, pred_u)

                tl += loss.item()

                optim.zero_grad()
                loss.backward()
                optim.step()
            if verbose:
                y_pred, y_test = self.decision_function_loader(test_loader)
                auc = metrics.roc_auc_score(y_test, y_pred)
                print(f"[{epoch}/{num_epochs}]: loss={tl:.4f} auc={auc:.4f}")
            scheduler.step()
