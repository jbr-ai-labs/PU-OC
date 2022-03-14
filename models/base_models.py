import os
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.optim as opt
from sklearn import metrics
from torch import nn

from classifiers import Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Classifier(nn.Module):
    def __init__(self, model=Net, neg_label=0):
        """
        :param model: neural classifier
        :param neg_label: label of negative class (normally 0, for SVM-like models -1)
        """
        super().__init__()
        self.model = model()
        self.optimizers = None
        self.schedulers = None
        self.neg_label = neg_label

        self.val_data = None
        self.epoch = -1

    def forward(self, x):
        return self.model(x)

    @abstractmethod
    def batch_loss(self, batch):
        """
        :return: loss on a batch
        """
        raise NotImplementedError()

    def preprocess_data(self, data):
        """
        Preprocession for some methods
        :param data: input data for preprocessing
        :return: preprocessed data,
        """
        return data

    def postprocess(self):
        """
        Postprocession for some methods.
        """
        pass

    @abstractmethod
    def get_data_loader(self, train_data, batch_size):
        """
        :return: iterator over data
        """
        raise NotImplementedError()

    def _decision_function(self, x):
        """
        :param x: input
        :return: decision function for x
        """
        return self.model(x)

    def decision_function_loader(self, dataloader):
        """
        :return: calculates decision function on batched data
        """
        y_pred = np.array([])
        y_true = np.array([])

        for (x, y, _) in dataloader:
            res = self._decision_function(x.to(device).float())
            y_pred = np.hstack((y_pred, res.squeeze().detach().cpu().numpy()))
            y_true = np.hstack((y_true, y.squeeze().detach().cpu().numpy()))

        return y_pred, y_true

    def decision_function(self, data):
        """
        :return: calculates decision function on non-batched data
        """
        dataloader = torch.utils.data.DataLoader(data,
                                                 batch_size=512,
                                                 shuffle=False)

        return self.decision_function_loader(dataloader)

    def get_optimizers(self, lr, **kwargs):
        self.optimizers = [opt.Adam(self.model.parameters(), lr=lr)]

    def optimizers_step(self, loss):
        for optim in self.optimizers:
            optim.zero_grad()
        loss.backward()

        for optim in self.optimizers:
            optim.step()

    def get_schedulers(self, gamma, **kwargs):
        self.schedulers = [opt.lr_scheduler.ExponentialLR(self.optimizers[0], gamma=gamma)]

    def schedulers_step(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def fit(self,
            train_data,
            num_epochs=50,
            lr=1e-3,
            batch_size=512,
            gamma=0.96,
            verbose=False,
            test_data=None,
            **kwargs):
        self.to(device)

        train_data = self.preprocess_data(train_data)
        data_loader = self.get_data_loader(train_data, batch_size)

        if test_data and verbose:
            test_loader = torch.utils.data.DataLoader(test_data,
                                                      batch_size=512,
                                                      shuffle=False)

        self.get_optimizers(lr, **kwargs)
        self.get_schedulers(gamma, **kwargs)

        for epoch in range(num_epochs):
            self.epoch += 1
            self.train()

            running_loss = 0.0

            for batch in data_loader:
                loss = self.batch_loss(batch)

                self.optimizers_step(loss)

                running_loss += loss.item()

            self.schedulers_step()
            if verbose:
                print_line = f'[{epoch}/{num_epochs}]: loss={running_loss:.4f}'

                if test_data is not None and verbose:
                    self.eval()
                    y_pred, y_true = self.decision_function_loader(test_loader)
                    auc = metrics.roc_auc_score(y_true, y_pred)
                    print_line += f" auc={auc:.4f}"
                    # if 'acc' in kwargs:
                    #     acc = metrics.roc_auc_score(y_true, y_pred > 0.5)
                    #     print_line += f" acc={acc:.4f}"

                print(print_line)
        self.postprocess()

    def save(self, path):
        torch.save(self.model.state_dict(), os.path.join(path, 'model.pt'))

    def load(self, path):
        self.model.load_state_dict(torch.load(os.path.join(path, 'model.pt')))


class OCModel(Classifier, ABC):
    """
    Class for OC models
    """

    def get_data_loader(self, train_data, batch_size):
        oc_data = train_data.lab_data(lab=1)

        data_loader = torch.utils.data.DataLoader(oc_data,
                                                  batch_size=batch_size,
                                                  shuffle=True)

        return data_loader


class PUModelEqualBatch(Classifier, ABC):
    """
    Class for PU models which sample data from labeled and unlabeled samples in equal batches
    """

    def get_data_loader(self, train_data, batch_size):
        lab_data = train_data.lab_data(lab=1)
        unl_data = train_data.lab_data(lab=self.neg_label)

        lab_loader = torch.utils.data.DataLoader(lab_data,
                                                 batch_size=batch_size,
                                                 shuffle=True)

        unl_loader = torch.utils.data.DataLoader(unl_data,
                                                 batch_size=batch_size,
                                                 shuffle=True)

        return zip(lab_loader, unl_loader)


class PUModelRandomBatch(Classifier, ABC):
    """
    Class for PU models which sample data from all available data
    """

    def get_data_loader(self, train_data, batch_size):
        data_loader = torch.utils.data.DataLoader(train_data,
                                                  batch_size=batch_size,
                                                  shuffle=True)

        return data_loader
