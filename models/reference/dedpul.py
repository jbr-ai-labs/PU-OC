import torch
from torch import nn


# code adapted from https://github.com/dimonenka/DEDPUL
from dedpul_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DEDPUL(nn.Module):
    def __init__(self, model):
        """
        :param model: any probabilistic pu classifier with fit method
        """
        super().__init__()
        self.model = model

    def fit(self,
            train_data,
            pre_trained=False,
            num_epochs=8,
            lr=1e-3,
            bw_mix=0.05,
            bw_pos=0.1,
            MT_coef=0.25,
            quantile=0.05,
            max_it=100,
            verbose=False,
            test_data=None):
        if not pre_trained:
            self.model.fit(train_data, num_epochs=num_epochs, verbose=verbose, test_data=None, lr=lr)

        preds, _ = self.model.decision_function(train_data)

        threshold_h = preds[train_data.s == 0].mean()
        threshold_l = preds[train_data.s == 1].mean()

        threshold = (threshold_h + threshold_l) / 2

        k_neighbours = train_data.data.shape[0] // 20

        diff_preds = np.minimum(1 - preds, 1 - 1e-8)
        diff_preds = np.maximum(diff_preds, 1e-8)

        diff = estimate_diff(diff_preds, 1 - train_data.s, bw_mix, bw_pos, 'logit', threshold, k_neighbours,
                             MT=True, MT_coef=MT_coef, tune=False)

        # test_alpha, poster = estimate_poster_dedpul(diff, quantile=quantile, max_it=max_it)
        test_alpha, poster = estimate_poster_em(diff, mode='dedpul', converge=True, nonconverge=True,
                                                max_diff=0.05, step=0.0025, alpha_as_mean_poster=True,
                                                quantile=quantile, max_it=max_it)

        # preds = preds[]

        preds = preds[(train_data.s == 0).numpy()]

        # sorted ntc(U)
        arg_preds = np.argsort(preds)
        sort_preds = preds[arg_preds]

        # ntc(test)
        test_preds, _ = self.model.decision_function(test_data)

        # lower bound for ntc(test)
        insert_ind = np.searchsorted(sort_preds, test_preds)

        # indecies of closest points
        lower = np.maximum(insert_ind - 1, 0)
        upper = np.minimum(insert_ind, len(sort_preds) - 1)

        # index of closest point
        # true_ind = test_preds - sort_preds[lower] > sort_preds[upper] - test_preds
        # inter_ind = lower
        # inter_ind[true_ind] = upper[true_ind]

        # unsorted index of closest point
        # inter_ind = arg_preds[inter_ind]

        # poster of closest point
        # test_poster = diff[inter_ind]
        # test_poster = 1 - poster[inter_ind]

        lower_poster = 1 - poster[arg_preds[lower]]
        upper_poster = 1 - poster[arg_preds[upper]]

        lower_dist = test_preds - sort_preds[lower]
        upper_dist = sort_preds[upper] - test_preds

        test_poster = (lower_dist * lower_poster + upper_dist * upper_poster) / (lower_dist + upper_dist + 1e-8)

        return test_alpha, (test_data.y, test_poster)
