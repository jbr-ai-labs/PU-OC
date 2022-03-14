import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as opt
from sklearn.svm import OneClassSVM

from models.base_models import PUModelRandomBatch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OC_SVM(OneClassSVM):
    def __init__(self, dim, nu=0.5, gamma=None, kernel="linear", **kwargs):
        if gamma is None:
            gamma = 0.2 / dim
        self.pu = False
        self.ae = True
        super().__init__(nu=nu, gamma=gamma, kernel=kernel)

    def fit(self, X, y=None, sample_weight=None, **params):
        lab_data = X.lab_data(lab=1)
        super().fit(np.array(lab_data.data))

    def decision_function(self, dataset):
        return super().decision_function(np.array(dataset.data))


class PU_SVM(PUModelRandomBatch):
    def __init__(self, dim, kernel="poly", gamma=1, degree=1, coef0=0.0, pi=0.5, lam=0.01, **kwargs):
        if kernel == "poly":
            self.w = torch.rand(dim, 1, requires_grad=True)
        elif kernel == "rbf":
            self.w = torch.rand(dim, requires_grad=True)
        else:
            raise ValueError(f"wrong kernel type = {kernel}")
        self.b = 1

        self.coef0 = coef0
        self.degree = degree
        self.kernel = kernel
        self.gamma = gamma
        self.dim = dim
        self.pi = pi
        self.lam = lam

        self.pu = True
        self.ae = True

        super().__init__(None, -1)

    def _decision_function(self, x, **kwargs):
        return self.forward(x)

    def forward(self, x, **kwargs):
        x = torch.Tensor(x)
        if self.kernel == "rbf":
            return torch.exp(-self.gamma * (torch.pow(torch.norm(x - self.w, dim=-1), 2))) - self.b
        elif self.kernel == "poly":
            return torch.pow(self.coef0 + self.gamma * torch.mm(x, self.w), self.degree) - self.b
        else:
            raise NotImplementedError(f'wrong kernel type = {self.kernel}')

    def get_optimizers(self, lr, **kwargs):
        self.optimizers = [opt.Adam([self.w], lr=lr)]

    def predict(self, x, **kwargs):
        return (self.decision_function(x) > 0) * 2 - 1

    def batch_loss(self, batch):
        batch_x, _, batch_s = batch
        batch_x = batch_x.to(device).float()

        preds = self.forward(batch_x)

        batch_p = batch_x[batch_s == 1]
        batch_u = batch_x[batch_s == self.neg_label]

        preds_p = preds[batch_s == 1]
        preds_u = preds[batch_s == self.neg_label]

        if len(batch_p):
            L_pos = F.relu(torch.max(1 - preds_p, -2 * preds_p)).mean()
            L_neg = -F.relu(torch.max(1 + preds_p, 2 * preds_p)).mean()
        else:
            L_pos = 0
            L_neg = 0

        if len(batch_u):
            L_unl = F.relu(torch.max(1 + preds_u, 2 * preds_u)).mean()
        else:
            L_unl = 0

        if L_unl + self.pi * L_neg > 0:
            L = self.lam * torch.pow(torch.norm(self.w), 2) + L_pos * self.pi + L_unl + self.pi * L_neg
        else:
            L = self.lam * torch.pow(torch.norm(self.w), 2) + L_pos * self.pi

        # 1
        self.b += np.percentile(preds_u.detach().cpu().numpy(), q=100 * (1 - self.pi))

        # 2
        # self.b += np.percentile(y_pred.detach().cpu().numpy(), q=100 * (1 - self.pi) / 2)

        loss = L - self.b

        return loss

# class EN_SVM:
#     def __init__(self, dim, gamma=None, kernel="rbf", **kwargs):
#         if gamma is None:
#             gamma = 2 / dim
#         self.pu = True
#         self.ae = True
#         self.c = None
#         self.pi = None
#         self.clf = SVC(gamma=gamma, kernel=kernel)
#
#     def run_train(self, dataset, test_size=0.2):
#         data = np.array(dataset.data)
#         targets = np.array(dataset.s)
#         # lab_portion = (targets == 1).sum() / len(targets)
#
#         data, val_data, targets, val_tar = train_test_split(data, targets, test_size=test_size)
#
#         lab, unl = targets == 1, targets == -1
#         lab_data, unl_data = data[lab], data[unl]
#
#         if len(lab_data) > len(unl_data):
#             lab_data = lab_data[:len(unl_data)]
#         else:
#             unl_data = unl_data[:len(lab_data)]
#
#         dataset = np.concatenate((lab_data, unl_data))
#         targets = np.concatenate((np.ones(len(lab_data)), np.zeros(len(unl_data))))
#
#         self.val = val_data, val_tar
#         self.clf.fit(dataset, targets)
#         self.clf_plt = CalibratedClassifierCV(self.clf, method="sigmoid").fit(dataset, targets)
#
#         self.c = self.clf_plt.predict_proba(val_data[val_tar == -1])[:, 1].max()
#         self.pi_est = (1 - self.c) / self.c
#
#     def predict(self, dataset, pi=None):
#         if pi is None:
#             pi = self.pi_est
#         return (self.decision_function(dataset.data) * pi > 0.5) * 2 - 1
#
#     def decision_function(self, dataset):
#         # return self.clf.decision_function(dataset)
#         res = self.clf_plt.predict_proba(dataset.data)[:, 1]
#         return res / (1 - res)
