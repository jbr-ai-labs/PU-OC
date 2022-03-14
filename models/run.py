import argparse
import pickle
from collections import defaultdict
from itertools import product

from sklearn import metrics

# from csi import CSI, PU_CSI
from datasets import *
from models.reference.drpu import DRPU
from old.en import EN
from models.reference.pan import PAN
from models.reference.repu import nnPU
from models.reference.vpu import VPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument('--lam', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=32, help='Training batch size.')
parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('--lr_lin', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('--gamma', type=float, default=0.96, help='lr decay')

parser.add_argument('--cls', type=int, default=None, help='pos cls index')
parser.add_argument('--mode', type=int, default=None, help='test or tune')
parser.add_argument('--shift', type=int, default=4, help='number of shifting transformations')
parser.add_argument('--simclr_dim', type=int, default=128, help='simclr dimension')
parser.add_argument('--model', type=str, default='csi_support', help='model')
parser.add_argument('--n', type=int, default=10, help='repeats')


def tune(model_cls,
         init_params,
         fit_params,
         train_holder=None,
         test_holder=None,
         pos_cls=None,
         alpha=0.5,
         size_lab=2500,
         size_unl=5000,
         repeats=5,
         splits=1,
         f=None):
    if pos_cls is None:
        pos_cls = [0, 1, 8, 9]
    result = defaultdict(lambda: [])
    result_d = defaultdict(lambda: [])

    if f is None:
        f = open(f'results/{model_cls.__name__}_results.txt', 'a')
    best_param = None
    best_auc = 0
    # best_param_d = None
    # best_auc_d = 0

    if train_holder is None:
        train_holder = get_data('CIFAR10', True, None, None, False)

    if test_holder is None:
        test_holder = get_data('CIFAR10', False, None, None, False)

    train_bin = train_holder.pos_neg_split(pos_cls)
    test_bin = test_holder.pos_neg_split(pos_cls)
    # params = {'shift_trans': Rotation(), 'K_shift': 4}

    init_keys = init_params.keys()
    fit_keys = fit_params.keys()

    for split in range(splits):
        for repeat in range(repeats):
            print(f"starting repeat {repeat + 1}", flush=True)
            print(f"starting repeat {repeat + 1}", flush=True, file=f)
            train_data, pi = train_bin.get_dataset(alpha, svm_labels=False, size_lab=size_lab, size_unl=size_unl)
            test_data, pi_test = test_bin.get_dataset(pi, c=0, svm_labels=False)

            for init_param in product(*init_params.values()):
                for fit_param in product(*fit_params.values()):
                    init_dict = {arg: val for (arg, val) in zip(init_keys, init_param)}
                    fit_dict = {arg: val for (arg, val) in zip(fit_keys, fit_param)}

                    model = model_cls(**init_dict)
                    model.fit(train_data, verbose=False, **fit_dict)

                    out = model.decision_function(test_data)

                    param = init_param + fit_param

                    result[param].append(metrics.roc_auc_score(out[1], out[0]))

                    # dedpul = DEDPUL(model)
                    # _, out = dedpul.fit(train_data, pre_trained=True, test_data=test_data)
                    # result_d[param].append(metrics.roc_auc_score(out[0], out[1]))

            for param in result:
                auc = np.mean(result[param])
                print({arg: val for (arg, val) in zip(list(init_keys) + list(fit_keys), param)}, flush=True)
                print({arg: val for (arg, val) in zip(list(init_keys) + list(fit_keys), param)}, flush=True, file=f)
                print(f"auc={auc:.4f}±{np.std(result[param])}", flush=True)
                print(f"auc={auc:.4f}±{np.std(result[param])}", flush=True, file=f)
            best_auc = 0
            best_param = None
            print(f"finished repeat {repeat + 1}", flush=True)
            print(f"finished repeat {repeat + 1}", flush=True, file=f)

    for param in result:
        auc = np.mean(result[param])
        if auc > best_auc:
            best_auc = auc
            best_param = param
        print({arg: val for (arg, val) in zip(list(init_keys) + list(fit_keys), param)}, flush=True)
        print(f"auc={auc:.4f}±{np.std(result[param])}", flush=True)

    # for param in result_d:
    #     auc = np.mean(result_d[param])
    #     if auc > best_auc_d:
    #         best_auc_d = auc
    #         best_param_d = param

    arg_dict = {arg: val for (arg, val) in zip(list(init_keys) + list(fit_keys), best_param)}
    print('-' * 10 + '\nBEST RESULTS', flush=True)
    print(f"best_auc={best_auc}", flush=True)
    print(arg_dict, flush=True)

    # arg_dict = {arg: val for (arg, val) in zip(list(init_keys) + list(fit_keys), best_param_d)}
    # print('-' * 10 + '\nBEST RESULTS', flush=True)
    # print(f"best_auc={best_auc_d}", flush=True)
    # print(arg_dict, flush=True)

    print(
        f'pos={pos_cls}: best auc={best_auc:.4f} best_params={arg_dict}',
        file=f, flush=True)

    # return result


# def main(args):
#     train_holder = get_data('CIFAR10', True, None, None, False)
#     test_holder = get_data('CIFAR10', False, None, None, False)
#
#     cls = args.cls
#     alpha = 0.5
#
#     train_bin = train_holder.pos_neg_split(cls)
#     test_bin = test_holder.pos_neg_split(cls)
#
#     train_data, pi = train_bin.get_dataset(alpha, c=0.5, svm_labels=False)
#     test_data, pi_test = test_bin.get_dataset(alpha, c=0, svm_labels=False)
#
#     if args.model == 'csi_support':
#         model = CSI(Net, lam=args.lam, simclr_dim=args.simclr_dim, k_shift=args.shifts)
#     elif args.model == 'en_csi':
#         model = EN_CSI(Net, lam=args.lam, simclr_dim=args.simclr_dim, k_shift=args.shifts)
#     elif args.model == 'pu_csi_u':
#         model = PU_CSI(Net, lam=args.lam, simclr_dim=args.simclr_dim, k_shift=args.shifts, flag=True)
#     elif args.model == 'pu_csi_pu':
#         model = PU_CSI(Net, lam=args.lam, simclr_dim=args.simclr_dim, k_shift=args.shifts, flag=False)
#     else:
#         raise ValueError('wrong model')
#
#     model.fit(train_data, num_epochs=args.num_epochs, lr=args.lr, test_data=test_data, verbose=True,
#               batch_size=args.batch_size,
#               gamma=args.gamma)


def hyper_tune(args):
    train_holder = get_data('CIFAR10', True, None, None, False)
    test_holder = get_data('CIFAR10', False, None, None, False)

    # init_params = {'lam': [0.25, 0.5, 0.75], 'simclr_dim': [128], 'K_shift': [4]}
    # fit_params = {'lr': [1e-3, 1e-4], 'num_epochs': [20, 50]}

    # for cls in ([None] + list(range(10))):
    #     print('-*20', flush=True)
    #     print(f'start tuning cls={cls}', flush=True)
    #     tune(CSI, init_params, fit_params, train_holder=train_holder, test_holder=test_holder, repeats=5)
    if args.model == 'csi_support':
        model = CSI
        init_params = {'lam': [0.25, 0.5, 0.75], 'simclr_dim': [128], 'K_shift': [4]}
        fit_params = {'lr': [1e-3, 1e-4], 'num_epochs': [20, 50]}
    elif args.model == 'pu_csi':
        model = PU_CSI
        init_params = {'lam': [0.25, 0.5, 0.75], 'simclr_dim': [128], 'K_shift': [4]}
        fit_params = {'lr': [1e-3, 1e-4], 'num_epochs': [20, 50]}
    elif args.model == 'vpu':
        model = VPU
        init_params = {'lam': [0.03, 0.1, 0.5], 'alpha': [0.01, 0.5]}
        fit_params = {'lr': [5e-3, 1e-4], 'num_epochs': [10, 50], 'gamma': [0.99]}
    elif args.model == 'drpu':
        model = DRPU
        init_params = {'alpha': [0.001, 0.1]}
        fit_params = {'lr': [1e-3, 1e-4], 'num_epochs': [20, 60], 'gamma': [0.99]}
    elif args.model == 'nnpu':
        model = nnPU
        init_params = {'alpha': [0.5]}
        fit_params = {'lr': [1e-3, 1e-4], 'num_epochs': [10, 20, 50], 'gamma': [0.99]}
    elif args.model == 'en':
        model = EN
        init_params = {}
        fit_params = {'lr': [1e-3, 1e-4], 'num_epochs': [10, 20, 50], 'gamma': [0.99]}
    elif args.model == 'pan':
        model = PAN
        init_params = {'lamda': [0.001, 0.1]}
        fit_params = {'lr': [1e-3, 1e-4], 'num_epochs': [10, 20, 50], 'gamma': [0.99]}
    else:
        raise ValueError('wrong model')
    tune(model, init_params, fit_params, train_holder=train_holder, test_holder=test_holder, pos_cls=args.cls,
         repeats=5)


def alpha_test(args):

    name = 'pu_alpha'
    params = {}
    params['PAN'] = ({'lamda': 0.001},  {'lr': 1e-3, 'num_epochs': 20, 'gamma': 0.99})
    params['VPU'] = ({'lam': 0.03, 'alpha': 0.01}, {'lr': 5e-3, 'num_epochs': 50, 'gamma': 0.99})
    params['DRPU'] = ({'alpha': 0.001}, {'lr': 1e-3, 'num_epochs': 60, 'gamma': 0.99})
    params['EN'] = ({},  {'lr': 1e-3, 'num_epochs': 20, 'gamma': 0.99})
    params['nnPU'] = ({},  {'lr': 1e-3, 'num_epochs': 20, 'gamma': 0.99})


    alphas = [0.05, 0.25, 0.5, 0.75, 0.95]
    n = args.n
    models_block = [PAN, VPU, DRPU]
    results = {}
    for model_type in models_block:
        results[model_type.__name__] = {cls: [] for cls in alphas}

    train_holder = get_data('CIFAR10',
                            True,
                            pos_label=None,
                            neg_label=None,
                            norm_flag=False)

    test_holder = get_data('CIFAR10',
                           False,
                           pos_label=None,
                           neg_label=None,
                           norm_flag=False)

    pos_cls = [0, 1, 8, 9]
    train_bin = train_holder.pos_neg_split(pos_cls)
    test_bin = test_holder.pos_neg_split(pos_cls)

    for i, alpha in enumerate(alphas):
        for _ in range(n):
            train_data, pi = train_bin.get_dataset(alpha, svm_labels=False, size_lab=2500, size_unl=5000)
            test_data, pi_test = test_bin.get_dataset(alpha, c=0, svm_labels=False)

            for model_type in models_block:
                model = model_type(**params[model_type.__name__][0])
                model.fit(train_data, verbose=False,**params[model_type.__name__][1])
                out = model.decision_function(test_data)
                results[model_type.__name__][alpha].append(metrics.roc_auc_score(out[1], out[0]))
            with open(f'results/{name}.pcl', 'wb') as f:
                pickle.dump(results, f)
            print(f'done alpha={alpha:.2f} n={n}', flush=True)

def ova_test(args):

    name = 'pu_ova'
    params = {}
    params['PAN'] = ({'lamda': 0.001},  {'lr': 1e-3, 'num_epochs': 20, 'gamma': 0.99})
    params['VPU'] = ({'lam': 0.03, 'alpha': 0.01}, {'lr': 5e-3, 'num_epochs': 50, 'gamma': 0.99})
    params['DRPU'] = ({'alpha': 0.001}, {'lr': 1e-3, 'num_epochs': 60, 'gamma': 0.99})


    alpha = 0.5
    pos_labels = np.arange(10)
    n = args.n
    models_block = [PAN, VPU, DRPU]
    results = {}
    for model_type in models_block:
        results[model_type.__name__] = {cls: [] for cls in pos_labels}

    train_holder = get_data('CIFAR10',
                            True,
                            pos_label=None,
                            neg_label=None,
                            norm_flag=False)

    test_holder = get_data('CIFAR10',
                           False,
                           pos_label=None,
                           neg_label=None,
                           norm_flag=False)

    # pos_cls = [0, 1, 8, 9]

    for i, pos_cls in enumerate(pos_labels):
        for j in range(n):
            train_bin = train_holder.pos_neg_split(pos_cls)
            test_bin = test_holder.pos_neg_split(pos_cls)
            train_data, pi = train_bin.get_dataset(alpha, svm_labels=False, size_lab=2500, size_unl=5000)
            test_data, pi_test = test_bin.get_dataset(alpha, c=0, svm_labels=False)

            for model_type in models_block:
                model = model_type(**params[model_type.__name__][0])
                model.fit(train_data, verbose=False,**params[model_type.__name__][1])
                out = model.decision_function(test_data)
                results[model_type.__name__][alpha].append(metrics.roc_auc_score(out[1], out[0]))
            with open(f'results/{name}.pcl', 'wb') as f:
                pickle.dump(results, f)
            print(f'done cls={pos_cls} n={j}', flush=True)

def negshift_test(args):

    name = 'pu_alpha'
    params = {}
    params['PAN'] = ({'lamda': 0.001},  {'lr': 1e-3, 'num_epochs': 20, 'gamma': 0.99})
    params['VPU'] = ({'lam': 0.03, 'alpha': 0.01}, {'lr': 5e-3, 'num_epochs': 50, 'gamma': 0.99})
    params['DRPU'] = ({'alpha': 0.001}, {'lr': 1e-3, 'num_epochs': 60, 'gamma': 0.99})


    alphas = [0.05, 0.25, 0.5, 0.75, 0.95]
    n = args.n
    models_block = [PAN, VPU, DRPU]
    results = {}
    for model_type in models_block:
        results[model_type.__name__] = {cls: [] for cls in alphas}

    train_holder = get_data('CIFAR10',
                            True,
                            pos_label=None,
                            neg_label=None,
                            norm_flag=False)

    test_holder = get_data('CIFAR10',
                           False,
                           pos_label=None,
                           neg_label=None,
                           norm_flag=False)

    pos_cls = [0, 1, 8, 9]
    train_bin = train_holder.pos_neg_split(pos_cls)
    test_bin = test_holder.pos_neg_split(pos_cls)

    for i, alpha in enumerate(alphas):
        for _ in range(n):
            train_data, pi = train_bin.get_dataset(alpha, svm_labels=False, size_lab=2500, size_unl=5000)
            test_data, pi_test = test_bin.get_dataset(alpha, c=0, svm_labels=False)

            for model_type in models_block:
                model = model_type(**params[model_type.__name__][0])
                model.fit(train_data, verbose=False,**params[model_type.__name__][1])
                out = model.decision_function(test_data)
                results[model_type.__name__][alpha].append(metrics.roc_auc_score(out[1], out[0]))
            with open(f'results/{name}.pcl', 'wb') as f:
                pickle.dump(results, f)
            print(f'done alpha={alpha:.2f} n={n}', flush=True)

def nonrnd_negshift_test(args):

    name = 'pu_alpha'
    params = {}
    params['PAN'] = ({'lamda': 0.001},  {'lr': 1e-3, 'num_epochs': 20, 'gamma': 0.99})
    params['VPU'] = ({'lam': 0.03, 'alpha': 0.01}, {'lr': 5e-3, 'num_epochs': 50, 'gamma': 0.99})
    params['DRPU'] = ({'alpha': 0.001}, {'lr': 1e-3, 'num_epochs': 60, 'gamma': 0.99})


    alphas = [0.05, 0.25, 0.5, 0.75, 0.95]
    n = args.n
    models_block = [PAN, VPU, DRPU]
    results = {}
    for model_type in models_block:
        results[model_type.__name__] = {cls: [] for cls in alphas}

    train_holder = get_data('CIFAR10',
                            True,
                            pos_label=None,
                            neg_label=None,
                            norm_flag=False)

    test_holder = get_data('CIFAR10',
                           False,
                           pos_label=None,
                           neg_label=None,
                           norm_flag=False)

    pos_cls = [0, 1, 8, 9]
    train_bin = train_holder.pos_neg_split(pos_cls)
    test_bin = test_holder.pos_neg_split(pos_cls)

    for i, alpha in enumerate(alphas):
        for _ in range(n):
            train_data, pi = train_bin.get_dataset(alpha, svm_labels=False, size_lab=2500, size_unl=5000)
            test_data, pi_test = test_bin.get_dataset(alpha, c=0, svm_labels=False)

            for model_type in models_block:
                model = model_type(**params[model_type.__name__][0])
                model.fit(train_data, verbose=False,**params[model_type.__name__][1])
                out = model.decision_function(test_data)
                results[model_type.__name__][alpha].append(metrics.roc_auc_score(out[1], out[0]))
            with open(f'results/{name}.pcl', 'wb') as f:
                pickle.dump(results, f)
            print(f'done alpha={alpha:.2f} n={n}', flush=True)

def abnormal_test(args):

    name = 'pu_alpha'
    params = {}
    params['PAN'] = ({'lamda': 0.001},  {'lr': 1e-3, 'num_epochs': 20, 'gamma': 0.99})
    params['VPU'] = ({'lam': 0.03, 'alpha': 0.01}, {'lr': 5e-3, 'num_epochs': 50, 'gamma': 0.99})
    params['DRPU'] = ({'alpha': 0.001}, {'lr': 1e-3, 'num_epochs': 60, 'gamma': 0.99})


    alphas = [0.05, 0.25, 0.5, 0.75, 0.95]
    n = args.n
    models_block = [PAN, VPU, DRPU]
    results = {}
    for model_type in models_block:
        results[model_type.__name__] = {cls: [] for cls in alphas}

    train_holder = get_data('CIFAR10',
                            True,
                            pos_label=None,
                            neg_label=None,
                            norm_flag=False)

    test_holder = get_data('CIFAR10',
                           False,
                           pos_label=None,
                           neg_label=None,
                           norm_flag=False)

    pos_cls = [0, 1, 8, 9]
    train_bin = train_holder.pos_neg_split(pos_cls)
    test_bin = test_holder.pos_neg_split(pos_cls)

    for i, alpha in enumerate(alphas):
        for _ in range(n):
            train_data, pi = train_bin.get_dataset(alpha, svm_labels=False, size_lab=2500, size_unl=5000)
            test_data, pi_test = test_bin.get_dataset(alpha, c=0, svm_labels=False)

            for model_type in models_block:
                model = model_type(**params[model_type.__name__][0])
                model.fit(train_data, verbose=False,**params[model_type.__name__][1])
                out = model.decision_function(test_data)
                results[model_type.__name__][alpha].append(metrics.roc_auc_score(out[1], out[0]))
            with open(f'results/{name}.pcl', 'wb') as f:
                pickle.dump(results, f)
            print(f'done alpha={alpha:.2f} n={n}', flush=True)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.mode == 1:
        hyper_tune(args)
    elif args.mode == 2:
        alpha_test(args)
    elif args.mode == 3:
        ova_test(args)
    elif args.mode == 4:
        negshift_test(args)
    elif args.mode == 5:
        alpha_test(args)
    elif args.mode == 6:
        abnormal_test(args)
    # elif args.mode == 0:
    #     main(args)
