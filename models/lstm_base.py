from itertools import chain

import sklearn.metrics as metrics
import torch.nn.functional as F
import torch.optim as opt
import torch.utils.data
import torch.utils.data

from datasets import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMbase:
    def __init__(self, vec_size=1, nu=0.5, hd=10):
        self.w = torch.rand(hd, 1, requires_grad=True, device=device)
        self.r = 1
        self.nu = nu
        self.lstm = torch.nn.LSTM(vec_size, hd, batch_first=True).to(device)

    def predict(self, test_data):
        testloader = torch.utils.data.DataLoader(test_data,
                                                 batch_size=512)

        y_true = np.array([])
        y_pred = np.array([])

        for (x_test, y_test, _, l_test) in testloader:
            lstm_out = self.lstm(x_test.to(device).float())
            hm = lstm_out[0][np.arange(len(x_test)), l_test - 1, :]
            res = -self.r + torch.mm(hm, self.w)

            y_true = np.hstack((y_true, y_test.squeeze().detach().cpu().numpy()))
            y_pred = np.hstack((y_pred, res.squeeze().detach().cpu().numpy()))
        return y_pred, y_true

    def loss(self, y_pred, y_true):
        raise NotImplemented()

    def get_pred(self, hm):
        raise NotImplemented()

    def run_train(self, train_data,
                  lr=1e-3, num_epochs=50,
                  batch_size=512, gamma=0.99,
                  test_data=None, mode="text", verbose=True):
        optim = opt.Adam(chain(self.lstm.parameters(), [self.w]), lr=lr)
        scheduler = opt.lr_scheduler.ExponentialLR(optim, gamma=gamma)

        dataloader = torch.utils.data.DataLoader(train_data,
                                                 batch_size=batch_size,
                                                 shuffle=True)

        for epoch in range(num_epochs):
            running_loss = 0.0
            for (x_batch, _, s_batch, l_batch) in dataloader:
                x_batch = x_batch.to(device)
                s_batch = s_batch.to(device)
                lstm_out = self.lstm(x_batch.float())
                if mode == "mean":
                    hm = torch.mean(lstm_out[0], axis=1)
                elif mode == "last":
                    hm = lstm_out[0][:, -1, :]
                elif mode == "text":
                    hm = lstm_out[0][np.arange(len(x_batch)), l_batch - 1, :]
                else:
                    raise ValueError(f"Wrong mode {mode}")

                y_pred = self.get_pred(hm)
                loss = self.loss(y_pred, s_batch)

                optim.zero_grad()
                loss.backward()
                optim.step()

                running_loss += loss.item()

            scheduler.step()

            if verbose:
                y_pred, y_true = self.predict(test_data)
                if (epoch + 1) % 1 == 0:
                    print(
                        f"[{epoch + 1:3}/{num_epochs}]: loss={running_loss:.4f} auc={metrics.roc_auc_score(y_true, y_pred):.4f}")


class OC_LSTM(LSTMbase):
    def get_pred(self, hm):
        y_pred = -torch.mm(hm, self.w) + self.r
        self.r = np.percentile(hm.detach().cpu().numpy(), q=100 * self.nu)
        return y_pred

    def loss(self, y_pred, y_true):
        return self.nu * torch.pow(torch.norm(self.w), 2) / 2 + torch.mean((torch.relu(y_pred)) - self.r)


class PU_LSTM(LSTMbase):
    def __init__(self, vec_size=1, nu=0.5, hd=10, pi=0.5):
        super().__init__(vec_size, nu, hd)
        self.pi = pi

    def loss(self, y_pred, y_true):
        pos_label = y_true == 1
        unl_label = y_true == 0

        pos_pred = y_pred[pos_label]
        unl_pred = y_pred[unl_label]

        L_pos = 0
        L_neg = 0
        L_unl = 0

        if len(pos_label):
            L_pos = F.relu(torch.max(1 - pos_pred, -2 * pos_pred)).mean()
            L_neg = -F.relu(torch.max(1 + pos_pred, 2 * pos_pred)).mean()

        if len(unl_label):
            L_unl = F.relu(torch.max(1 + unl_pred, 2 * unl_pred)).mean()

        if L_unl + self.pi * L_neg > 0:
            L = self.nu * torch.pow(torch.norm(self.w), 2) + L_pos * self.pi + L_unl + self.pi * L_neg
        else:
            L = self.nu * torch.pow(torch.norm(self.w), 2) + L_pos * self.pi

        self.r += np.percentile(y_pred.detach().cpu().numpy(), q=100 * (1 - self.pi))

        return L

    def get_pred(self, hm):
        y_pred = torch.mm(hm, self.w) - self.r
        return y_pred


# class EN_LSTM(LSTMbase):
#     def loss(self, y_pred, y_true):
#         pos_label = y_true == 1
#         unl_label = y_true == 0
#
#         pos_pred = y_pred[pos_label]
#         unl_pred = y_pred[unl_label]
#
#         L_pos = 0
#         L_unl = 0
#
#         if len(pos_label):
#             L_pos = F.relu(1 - pos_pred).mean()
#
#         if len(unl_label):
#             L_unl = F.relu(1 + unl_pred).mean()
#
#         L = L_pos + L_unl + self.nu * torch.pow(torch.norm(self.w), 2)
#
#         self.r = np.mean(y_pred.detach().cpu().numpy())
#
#         return L
#
#     def get_pred(self, hm):
#         y_pred = torch.mm(hm, self.w) - self.r
#         return y_pred