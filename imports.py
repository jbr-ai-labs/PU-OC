import sklearn.metrics as metrics
import torch.utils.data
from tqdm.notebook import tqdm

from datasets import *
from models.caenb import get_encoder
from tice import *


def run_test(models_list, data, pi, alpha=None):
    if alpha is None:
        alpha = pi
    result = {}
    pu_data = data[0]
    oc_data = data[1]

    for model in models_list:
        result[f"{type(model).__name__}"] = {}
        data = oc_data

        res = model.predict(data, pi=pi)
        alpha_est = (res == 1).sum() / len(res)
        result[f"{type(model).__name__}"]["auc"] = (metrics.roc_auc_score(data.y, model.decision_function(data)))
    return result


def encode_dataset(encoder, dataset):
    imloader = torch.utils.data.DataLoader(dataset.data,
                                           batch_size=512)

    new_x = None
    for batch in imloader:
        batch_encoded = encoder(batch).detach().cpu()
        if new_x is None:
            new_x = batch_encoded
        else:
            new_x = torch.cat((new_x, batch_encoded), dim=0)

    return Dataset(new_x, dataset.y, dataset.s)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
