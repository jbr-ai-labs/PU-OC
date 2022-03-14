from imports import *
from models.drocc import DROCC, PU_DROCC


def test_negshif(dataset_name,
                 models_block,
                 dim=128,
                 alpha=0.5,
                 n=1,
                 pos_label=0,
                 norm_flag=False,
                 cntrs=4,
                 params=None,
                 name='blank'):
    results = {}
    for model_type in models_block:
        results[model_type.__name__] = []

    train_holder = get_data(dataset_name,
                            True,
                            pos_label=None,
                            neg_label=None,
                            norm_flag=norm_flag)

    test_holder = get_data(dataset_name,
                           False,
                           pos_label=None,
                           neg_label=None,
                           norm_flag=norm_flag)

    for i in tqdm(range(cntrs)):
        # print(f"pos={pos_label}")
        # encoder = get_encoder(f'nbae_{pos_label}.pcl', dataset_name)

        # train_bin.encode_data(encoder)
        # test_bin.encode_data(encoder)

        res = {}
        for model_type in models_block:
            res[model_type.__name__] = []

        for _ in tqdm(range(n)):
            negs = np.random.choice(np.arange(1, 10), 2 * i + 2, replace=False)

            train_bin = train_holder.pos_neg_split([pos_label], negs[:i + 1])
            test_bin = train_holder.pos_neg_split([pos_label], negs[i + 1:])

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
