from imports import *
from models.drocc import DROCC, PU_DROCC


def test_multipos(alpha=0.5,
                  n=10,
                  pos_labels=[0, 1, 8, 9],
                  params=None,
                  name='blank'):
    results = {}

    models = ['DROCC', 'PU_DROCC']
    for model_type in models:
        results[model_type] = []

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

    for j in tqdm(range(len(pos_labels) - 1, len(pos_labels))):
        # print(f"pos={pos_label}")

        pos_cur = np.random.choice(pos_labels, size=j + 1, replace=False)

        train_bin = train_holder.pos_neg_split(pos_cur, [0, 1, 8, 9])
        test_bin = train_holder.pos_neg_split(pos_cur, [0, 1, 8, 9])

        res = {}
        for model_type in models:
            res[model_type] = []

        for i in tqdm(range(n)):
            train_data, pi = train_bin.get_dataset(alpha, svm_labels=False, size_lab=2500, size_unl=5000)
            test_data, pi_test = test_bin.get_dataset(alpha, c=0, svm_labels=False, size_unl=5000)

            test_loader = torch.utils.data.DataLoader(test_data,
                                                      batch_size=512,
                                                      shuffle=True)

            drocc = DROCC().to(device)
            drocc.run_train(train_data.lab_data(lab=1), None, **params[0]['DROCC'])
            res['DROCC'].append(drocc.test(test_loader))

            pu_drocc = PU_DROCC().to(device)
            pu_drocc.run_train(train_data, None, **params[0]['PU_DROCC'])
            res['PU_DROCC'].append(pu_drocc.test(test_loader))

        for model in res:
            results[model].append(res[model])

        # with open(f'/content/gdrive/My Drive/results/{name}.pcl', 'wb') as f:
        #     pickle.dump(results, f)

    return results
