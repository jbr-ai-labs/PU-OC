from itertools import product
from collections import defaultdict

import numpy as np
from sklearn import metrics
from tqdm import tqdm

from datasets import get_data


def tune(model_cls,
         init_params,
         fit_params,
         train_holder=None,
         test_holder=None,
         pos_cls=[0, 1, 8, 9],
         alpha=0.5,
         size_lab=3000,
         size_unl=6000,
         repeats=5,
         splits=1):
    result = defaultdict(lambda: [])
    best_param = None
    best_auc = 0

    if train_holder is None:
        train_holder = get_data('CIFAR10', True, None, None, False)
        test_holder = get_data('CIFAR10', False, None, None, False)

    train_bin = train_holder.pos_neg_split(pos_cls)
    test_bin = test_holder.pos_neg_split(pos_cls)

    init_keys = init_params.keys()
    fit_keys = fit_params.keys()

    for _ in range(splits):
        for _ in tqdm(range(repeats)):
            train_data, pi = train_bin.get_dataset(alpha, svm_labels=False, size_lab=size_lab, size_unl=size_unl)
            test_data, pi_test = test_bin.get_dataset(pi, c=0, svm_labels=False)

            for init_param in product(*init_params.values()):
                for fit_param in tqdm(product(*fit_params.values())):
                    init_dict = {arg: val for (arg, val) in zip(init_keys, init_param)}
                    fit_dict = {arg: val for (arg, val) in zip(fit_keys, fit_param)}

                    model = model_cls(**init_dict)
                    model.fit(train_data, verbose=False, **fit_dict)

                    out = model.decision_function(test_data)

                    param = init_param + fit_param

                    result[param].append(metrics.roc_auc_score(out[1], out[0]))

    for param in result:
        auc = np.mean(result[param])
        if auc > best_auc:
            best_auc = auc
            best_param = param

    print(f"best_auc={best_auc}")
    print({arg: val for (arg, val) in zip(list(init_keys) + list(fit_keys), best_param)})
    return result