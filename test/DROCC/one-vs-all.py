from imports import *
from models.drocc import PU_DROCC, DROCC


def test_onevsall(alpha=0.5,
                  n = 10,
                  pos_labels=[0],
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

    for pos_label in tqdm(pos_labels):

        train_bin = train_holder.pos_neg_split([pos_label])
        test_bin = train_holder.pos_neg_split([pos_label])



        res = {}
        for model_type in models:
            res[model_type] = []

        for i in tqdm(range(n)):
            train_data, pi = train_bin.get_dataset(alpha, svm_labels=False)
            test_data, pi_test = test_bin.get_dataset(alpha, c=0, svm_labels=False)

            test_loader = torch.utils.data.DataLoader(test_data,
                                                      batch_size=512,
                                                      shuffle=True)

            drocc = DROCC().to(device)
            drocc.run_train(train_data.lab_data(lab=1), None, **params[pos_label]['DROCC'])
            res['DROCC'].append(drocc.test(test_loader))

            pu_drocc = PU_DROCC().to(device)
            pu_drocc.run_train(train_data, None, **params[pos_label]['PU_DROCC'])
            res['PU_DROCC'].append(pu_drocc.test(test_loader))

        for model in res:
            results[model].append(res[model])

        # with open(f'/content/gdrive/My Drive/results/{name}.pcl', 'wb') as f:
        #     pickle.dump(results, f)

    return results
