from imports import *

def test_multipos(dataset_name,
                  models_block,
                  dim=128,
                  alpha=0.5,
                  n=1,
                  pos_labels=[0, 1, 8, 9],
                  norm_flag=False,
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

    for j in tqdm(range(0, len(pos_labels))):
        # for j in range(1):
        # print(f"pos={pos_label}")

        pos_cur = np.random.choice(pos_labels, size=j + 1, replace=False)
        enc = 'nbae_0189'
        encoder = get_encoder(f'nbae_anima.pcl', dataset_name)

        train_bin = train_holder.pos_neg_split(pos_cur, [0, 1, 8, 9])
        test_bin = train_holder.pos_neg_split(pos_cur, [0, 1, 8, 9])

        if svm:
            train_bin.encode_data(encoder)
            test_bin.encode_data(encoder)

        res = {}
        for model_type in models_block.values():
            res[model_type.__name__] = []

        for i in tqdm(range(n)):
            train_data, pi = train_bin.get_dataset(alpha, svm_labels=True, size_lab=2500, size_unl=5000)
            test_data, pi_test = test_bin.get_dataset(alpha, c=0, svm_labels=True, size_unl=5000)

            ntc_model = models_block["ntc"](dim=dim, encoder=enc, **params[0]['ntc']['init'])
            ntc_model.run_train(train_data, **params[0]['ntc']['train'])
            # ntc_model.eval()

            if not svm:
                encoded_train = encode_dataset(encoder, train_data)
                cest, _ = tice(encoded_train.data, encoded_train.s.numpy(), 1, np.random.randint(5, size=len(encoded_train.data)), delta=0.3)
                aest = (encoded_train.s == 1).sum() * (1-cest) / cest / (encoded_train.s != 1).sum()
            else:
                cest, _ = tice(train_data.data, train_data.s.numpy(), 3, np.random.randint(5, size=len(train_data.data)), delta=0.2)
                aest = (train_data.s == 1).sum() * (1-cest) / cest / (train_data.s != 1).sum()
            pi_est = min(1, aest)


            i = 0
            cur_models = [models_block["oc"](dim=dim, encoder=enc, **params[i]['oc']['init']),
                          models_block["pu"](dim=dim, pi=pi_est, encoder=enc, **params[0]['pu']['init'])]

            # cur_models = [models_block["oc"](dim=dim),
            #               models_block["pu"](dim=dim)]

            cur_models[0].run_train(train_data, **params[i]['oc']['train'])
            cur_models[1].run_train(train_data, **params[0]['pu']['train'])

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

        with open(f'/content/gdrive/My Drive/results/{name}.pcl', 'wb') as f:
            pickle.dump(results, f)

    return results
