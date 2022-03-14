from imports import *


def test_onevsall(dataset_name,
                  models_block,
                  dim=128,
                  alpha=0.5,
                  n=1,
                  norm_flag=False,
                  pos_labels=[0],
                  params=None,
                  svm=False,
                  name='blank'):
    results = {}
    for model_type in models_block.values():
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

    for pos_label in tqdm(pos_labels):
        # print(f"pos={pos_label}")
        encoder = get_encoder(f'nbae_{pos_label}.pcl', dataset_name)

        train_bin = train_holder.pos_neg_split([pos_label])
        test_bin = train_holder.pos_neg_split([pos_label])

        if svm:
            train_bin.encode_data(encoder)
            test_bin.encode_data(encoder)

        res = {}
        for model_type in models_block.values():
            res[model_type.__name__] = []

        for i in tqdm(range(n)):
            train_data, pi = train_bin.get_dataset(alpha, svm_labels=False)
            test_data, pi_test = test_bin.get_dataset(alpha, c=0, svm_labels=False)

            ntc_model = models_block["ntc"](dim=dim, encoder=f'nbae_{pos_label}', )
            ntc_model.run_train(train_data, )
            ntc_model.eval()

            if not svm:
                encoded_train = encode_dataset(encoder, train_data)
                cest, _ = tice(encoded_train.data, encoded_train.s.numpy(), 1,
                               np.random.randint(5, size=len(encoded_train.data)), delta=0.3)
                aest = (encoded_train.s == 1).sum() * (1 - cest) / cest / (encoded_train.s != 1).sum()
            else:
                cest, _ = tice(train_data.data, train_data.s.numpy(), 3,
                               np.random.randint(5, size=len(train_data.data)), delta=0.2)
                aest = (train_data.s == 1).sum() * (1 - cest) / cest / (train_data.s != 1).sum()
            pi_est = min(1, aest)

            cur_models = [models_block["oc"](dim=dim, encoder=f'nbae_{pos_label}', **params[pos_label]['oc']['init']),
                          models_block["pu"](dim=dim, pi=pi_est, encoder=f'nbae_{pos_label}',
                                             **params[pos_label]['pu']['init'])]

            # cur_models = [models_block["oc"](dim=dim),
            #               models_block["pu"](dim=dim)]

            cur_models[0].run_train(train_data, **params[pos_label]['oc']['train'])
            cur_models[1].run_train(train_data, **params[pos_label]['pu']['train'])

            # cur_models[0].run_train(train_data)
            # cur_models[1].run_train(train_data)

            test_models = [ntc_model,
                           cur_models[0],
                           cur_models[1]]
            # test_models = [cur_models[0]]

            # print(f"done training {i}")
            test = test_data, test_data

            result = run_test(test_models, test, None)

            for model in result:
                res[model].append(result[model]["auc"])
        for model in result:
            results[model].append(res[model])

    return results
