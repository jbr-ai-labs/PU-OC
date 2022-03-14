from imports import *
from models.drocc import DROCC
from models.svm_base import OC_SVM
from reference.pu import PU, EN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_unl_size(models_block,
                  alpha=0.95,
                  n=1,
                  size_lab=None,
                  sizes_unl=None,
                  params=None,
                  name='blank'):
    results = {}
    for model_type in models_block:
        results[model_type.__name__] = [[] for _ in sizes_unl]
    results['alpha'] = [[] for _ in sizes_unl]
    # results['PU_cor'] =  [[] for _ in sizes_unl]

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

    encoder = get_encoder(f'nbae_0189.pcl', 'CIFAR10')

    train_bin = train_holder.pos_neg_split([2, 3, 4, 5, 6, 7])
    test_bin = train_holder.pos_neg_split([2, 3, 4, 5, 6, 7])

    for _ in tqdm(range(n)):
        train_data, pi = train_bin.get_dataset(alpha, svm_labels=False, size_lab=size_lab)
        test_data, pi_test = test_bin.get_dataset(alpha, c=0, svm_labels=False)

        unl_size = (train_data.s == 0).sum()

        train_data_svm = train_data.encoder(encoder)
        train_data_svm.relabel(1, True)

        test_data_svm = test_data.encoder(encoder)
        test_data_svm.relabel(1, True)

        drocc = DROCC().to(device)
        drocc.run_train(train_data.lab_data(lab=1), None)

        test_loader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=512,
                                                  shuffle=True)
        drocc_res = drocc.test(test_loader)

        oc_svm = OC_SVM(128, kernel='linear')
        oc_svm.run_train(train_data_svm)
        oc_svm_res = metrics.roc_auc_score(test_data_svm.y, oc_svm.decision_function(test_data_svm))

        for i, size_unl in tqdm(enumerate(sizes_unl)):

            if i < len(sizes_unl) - 1:
                train_data_crop = train_data.subsample_unlabeled(size_unl)
            else:
                train_data_crop = train_data

            encoded_train = encode_dataset(encoder, train_data_crop)
            cest, _ = tice(encoded_train.data, encoded_train.s.numpy(), 1,
                           np.random.randint(5, size=len(encoded_train.data)), delta=0.3)
            aest = (encoded_train.s == 1).sum() * (1 - cest) / cest / (encoded_train.s != 1).sum()
            pi_est = min(1, aest)

            dedpul = DEDPUL().to(device)
            pi_est, dedpul_res = dedpul.run_train(train_data_crop, test_data)

            pu = PU().to(device)
            pu_res = pu.run_train(train_data_crop, test_data=test_data, alpha=pi_est)

            # pu = PU().to(device)
            # pu_res_cor = pu.run_train(train_data_crop, test_data=test_data, alpha=alpha)

            en = EN().to(device)
            en_res = en.run_train(train_data_crop, test_data=test_data)

            results['DROCC'][i].append(drocc_res)
            results['OC_SVM'][i].append(oc_svm_res)
            results['DEDPUL'][i].append(dedpul_res)
            results['PU'][i].append(pu_res)
            results['EN'][i].append(en_res)
            results['alpha'][i].append(pi_est)
            # results['PU_cor'][i].append(pu_res_cor)

            # with open(f'/content/gdrive/My Drive/results/{name}.pcl', 'wb') as f:
            #     pickle.dump(results, f)

    return results


def test_unl_alpha(models_block,
                   alphas=[0.5],
                   n=1,
                   size_lab=2500,
                   size_unl=5000,
                   params=None,
                   norm_flag=False,
                   svm_label=True,
                   name='blank'):
    results = {}
    for model_type in models_block.keys():
        results[model_type] = [[] for _ in alphas]

    results['alpha'] = [[] for _ in alphas]

    train_holder = get_data('CIFAR10',
                            True,
                            pos_label=None,
                            neg_label=None,
                            norm_flag=norm_flag)

    test_holder = get_data('CIFAR10',
                           False,
                           pos_label=None,
                           neg_label=None,
                           norm_flag=norm_flag)

    encoder = get_encoder(f'nbae_0189.pcl', 'CIFAR10')

    train_bin = train_holder.pos_neg_split([0, 1, 8, 9])
    test_bin = train_holder.pos_neg_split([0, 1, 8, 9])

    train_bin.encode_data(encoder)
    test_bin.encode_data(encoder)

    for i, alpha in tqdm(enumerate(alphas)):
        for _ in tqdm(range(n)):
            train_data, pi = train_bin.get_dataset(alpha, svm_labels=False, size_lab=size_lab)
            test_data, pi_test = test_bin.get_dataset(alpha, c=0, svm_labels=False)

            unl_size = (train_data.s == 0).sum()

            train_data_svm = train_data.encoder(encoder)
            train_data_svm.relabel(1, True)

            test_data_svm = test_data.encoder(encoder)
            test_data_svm.relabel(1, True)

            drocc = DROCC().to(device)
            drocc.run_train(train_data.lab_data(lab=1), None)

            test_loader = torch.utils.data.DataLoader(test_data,
                                                      batch_size=512,
                                                      shuffle=True)
            drocc_res = drocc.test(test_loader)

            oc_svm = OC_SVM(128, kernel='linear')
            oc_svm.run_train(train_data_svm)
            oc_svm_res = metrics.roc_auc_score(test_data_svm.y, oc_svm.decision_function(test_data_svm))

            encoded_train = encode_dataset(encoder, train_data)
            cest, _ = tice(encoded_train.data, encoded_train.s.numpy(), 1,
                           np.random.randint(5, size=len(encoded_train.data)), delta=0.3)
            aest = (encoded_train.s == 1).sum() * (1 - cest) / cest / (encoded_train.s != 1).sum()
            pi_est = min(1, aest)

            dedpul = DEDPUL().to(device)
            pi_est, dedpul_res = dedpul.run_train(train_data, test_data)

            pu = PU().to(device)
            pu_res = pu.run_train(train_data, test_data=test_data, alpha=pi_est)

            # pu = PU().to(device)
            # pu_res_cor = pu.run_train(train_data_crop, test_data=test_data, alpha=alpha)

            en = EN().to(device)
            en_res = en.run_train(train_data, test_data=test_data)

            results['DROCC'][i].append(drocc_res)
            results['OC_SVM'][i].append(oc_svm_res)
            results['DEDPUL'][i].append(dedpul_res)
            results['PU'][i].append(pu_res)
            results['EN'][i].append(en_res)
            results['alpha'][i].append(pi_est)
            # with open(f'/content/gdrive/My Drive/results/{name}.pcl', 'wb') as f:
            #     pickle.dump(results, f)

    return results
