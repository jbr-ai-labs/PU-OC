import os
import random
import zipfile

import numpy as np
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self, data, y, s):
        if isinstance(data, list):
            self.data = torch.stack(data)
            self.y = torch.Tensor(y)
            self.s = torch.Tensor(s)
        else:
            self.data = data.clone()
            self.y = y.clone()
            self.s = s.clone()

    def lab_data(self, lab=1):
        return Dataset(self.data[self.s == lab],
                       self.y[self.s == lab],
                       self.s[self.s == lab])

    def shift_data(self, pos=0, neg=1):
        return Dataset(self.data[(self.y == neg) + (self.y == pos)],
                       self.y[(self.y == neg) + (self.y == pos)],
                       self.s[(self.y == neg) + (self.y == pos)])

    def fix_label(self):
        self.y = (self.y + 1) / 2
        self.s = (self.s + 1) / 2

    def relabel(self, pos, svm=True):
        pos_ind = self.y == pos
        neg_ind = self.y != pos
        self.y[pos_ind] = 1
        if svm:
            self.y[neg_ind] = -1
        else:
            self.y[neg_ind] = 0

    def __getitem__(self, i):
        target, label = self.y[i], self.s[i]
        res = self.data[i]
        return res, target, label

    def __len__(self):
        return len(self.data)

    def subsample_unlabeled(self, size):
        lab_ind = self.s == 1
        unl_ind = self.s != 1

        subsample_ind = np.random.choice(np.arange(len(self.s))[unl_ind], size=size, replace=False)

        subsample = np.zeros(len(self.s), dtype=bool)
        subsample[lab_ind] = 1
        subsample[subsample_ind] = 1

        return Dataset(self.data[subsample],
                       self.y[subsample],
                       self.s[subsample])

    def encoder(self, encoder):
        new_data = encode_data(encoder, self.data, )
        return Dataset(new_data, self.y, self.s)


class DataHolder:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def pos_neg_split(self, pos_label=None, neg_label=None):
        if isinstance(pos_label, int):
            pos_label = [pos_label]

        pos = []
        neg = []

        for x, y in zip(self.data, self.labels):
            if y in pos_label:
                pos.append(x)
            elif neg_label is None or y in neg_label:
                neg.append(x)

        return BinaryDataHolder(pos, neg)


def encode_data(encoder, data):
    imloader = torch.utils.data.DataLoader(data,
                                           batch_size=512)

    new_data = None
    for batch in imloader:
        batch_encoded = encoder(batch).detach().cpu()
        if new_data is None:
            new_data = batch_encoded
        else:
            new_data = torch.cat((new_data, batch_encoded), dim=0)

    return new_data


class BinaryDataHolder:
    def __init__(self, pos, neg):
        self.pos = pos
        self.neg = neg

    def encode_data(self, encoder):
        self.pos = encode_data(encoder, self.pos)
        self.neg = encode_data(encoder, self.neg)

    def get_dataset(self, alpha=None, c=0.5, svm_labels=True, size_lab=None, size_unl=None):
        lab_pos, unl_pos = label_data(self.pos, c)
        unl_neg = list(self.neg)

        if size_lab:
            random.shuffle(lab_pos)
            lab_pos = lab_pos[:size_lab]

        if alpha:
            cur_alpha = len(unl_pos) / (len(unl_neg) + len(unl_pos))
            if cur_alpha > alpha:
                new_pos = int(alpha / (1 - alpha) * len(unl_neg))
                random.shuffle(unl_pos)
                unl_pos = unl_pos[:new_pos]
            else:
                new_neg = int((1 - alpha) / alpha * len(unl_pos))
                random.shuffle(unl_neg)
                unl_neg = unl_neg[:new_neg]

        prior = len(unl_pos) / (len(unl_pos) + len(unl_neg))

        if size_unl:
            size_unl_pos = int(prior * size_unl)
            size_unl_neg = size_unl - size_unl_pos

            random.shuffle(unl_pos)
            unl_pos = unl_pos[:size_unl_pos]

            random.shuffle(unl_neg)
            unl_neg = unl_neg[:size_unl_neg]

        # test_data = Dataset(test_pos + test_neg,
        #                     np.hstack((np.ones(len(test_pos)), np.zeros(len(test_neg)))),
        #                     np.zeros(len(test_pos) + len(test_neg)))

        if svm_labels:
            y_labels = np.concatenate((np.ones(len(lab_pos) + len(unl_pos)), -np.ones(len(unl_neg))))
            s_labels = np.concatenate((np.ones(len(lab_pos)), -np.ones(len(unl_pos) + len(unl_neg))))
        else:
            y_labels = np.concatenate((np.ones(len(lab_pos) + len(unl_pos)), np.zeros(len(unl_neg))))
            s_labels = np.concatenate((np.ones(len(lab_pos)), np.zeros(len(unl_pos) + len(unl_neg))))

        data = Dataset(lab_pos + unl_pos + unl_neg,
                       y_labels,
                       s_labels)

        return data, prior


def make_synthetic_data(pos_dist, neg_dist, alpha=0.5, n=5000):
    bern = np.random.rand(n) < alpha
    data = pos_dist.rvs(n) * bern[:, np.newaxis] + (1 - bern)[:, np.newaxis] * neg_dist.rvs(n)
    labels = np.ones(n) * bern - np.ones(n) * (1 - bern)
    return list(zip(data, labels))


def make_abnormal(drive, cat="Aeroplane", download=False):
    if download:
        data_file = drive.CreateFile({'id': "1S3zrwv0JpSq_CQ_ajnS105X9iQrzHIJQ"})
        data_file.GetContentFile('abnormal.zip')
        data_file = drive.CreateFile({'id': "1wk5bCJh5P7rsTQExitgiDG_XicQVIi1K"})
        data_file.GetContentFile('voc.zip')
        with zipfile.ZipFile("voc.zip", 'r') as zip_ref:
            zip_ref.extractall("")
        with zipfile.ZipFile("abnormal.zip", 'r') as zip_ref:
            zip_ref.extractall("")

    transform = transforms.Compose([transforms.Resize([200, 200]), transforms.ToTensor()])

    data = []
    ims_abnorm = os.listdir(f"abnormal/{cat}")
    ims_norm = os.listdir(f"voc/{cat}")
    for im in ims_abnorm:
        if im.endswith(".jpg"):
            img = transform(Image.open(f"abnormal/{cat}/{im}"))
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            data.append((img, 0))
    for im in ims_norm:
        if im.endswith(".jpg"):

            img = transform(Image.open(f"voc/{cat}/{im}"))
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            data.append((img, 1))

    train_data, test_data = train_test_split(data, test_size=0.2)
    return train_data, test_data


def get_data(dataset_name="MNIST",
             train=True,
             pos_label=None,
             neg_label=None,
             norm_flag=False,
             ):
    if dataset_name == "MNIST":
        if train:
            dataset = torchvision.datasets.MNIST("./data", train=True,
                                                 transform=transforms.ToTensor(),
                                                 download=True)
        else:
            dataset = torchvision.datasets.MNIST("./data", train=False,
                                                 transform=transforms.ToTensor(),
                                                 download=True)
    elif dataset_name == "CIFAR10":
        if train:
            dataset = torchvision.datasets.CIFAR10("./data", train=True,
                                                   transform=transforms.ToTensor(),
                                                   download=True)
        else:
            dataset = torchvision.datasets.CIFAR10("./data", train=False,
                                                   transform=transforms.ToTensor(),
                                                   download=True)
    elif dataset_name == "Abnormal":
        transform = transforms.ToTensor()
        if train:
            lib = f"dataset/train/{pos_label}"
        else:
            lib = f"dataset/test/{pos_label}"
        imgs = os.listdir(lib)
        dataset = []
        for im in imgs:
            if im.startswith("N"):
                label = 0
            else:
                label = 1
            dataset.append((transform(Image.open(f"{lib}/{im}")), label))
    else:
        raise ValueError("Not supported dataset. Choose from MNIST, CIFAR10, Abnormal")

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

    data = []
    labels = []

    for d in dataset:
        data.append(d[0] if not norm_flag else norm(d[0]))
        labels.append(d[1])

    return DataHolder(data, labels)


def make_data(dataset_name="MNIST",
              train=True,
              pos_label=None,
              alpha=None,
              neg_label=None,
              norm_flag=False,
              return_prior=False,
              c=0.5,
              **kwargs):
    """
        single training set
        :param norm_flag:
        :param train: return train or test dataset
        :param dataset_name: name of dataset: MNIST, CIFAR10, Abnormal, Synthetic
        :param pos_label: labels witch are considered as positive examples
        :param alpha: example probability in dataset
        :param neg_label: negative examples labels
        :param return_prior: returns prior probability
        :param c: label probability
    """
    if dataset_name == "MNIST":
        if train:
            dataset = torchvision.datasets.MNIST("./data", train=True,
                                                 transform=transforms.ToTensor(),
                                                 download=True)
        else:
            dataset = torchvision.datasets.MNIST("./data", train=False,
                                                 transform=transforms.ToTensor(),
                                                 download=True)
    elif dataset_name == "CIFAR10":
        if train:
            dataset = torchvision.datasets.CIFAR10("./data", train=True,
                                                   transform=transforms.ToTensor(),
                                                   download=True)
        else:
            dataset = torchvision.datasets.CIFAR10("./data", train=False,
                                                   transform=transforms.ToTensor(),
                                                   download=True)
    elif dataset_name == "Abnormal":
        transform = transforms.ToTensor()
        if train:
            lib = f"dataset/train/{pos_label}"
        else:
            lib = f"dataset/test/{pos_label}"
        imgs = os.listdir(lib)
        dataset = []
        for im in imgs:
            if im.startswith("N"):
                label = 0
            else:
                label = 1
            dataset.append((transform(Image.open(f"{lib}/{im}")), label))
        pos_label = [1]
    else:
        raise ValueError("Not supported dataset. Choose from MNIST, CIFAR10, Abnormal, Synthetic")

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

    if isinstance(pos_label, int):
        pos_label = [pos_label]

    def pos_neg_split(data, neg_label=None):
        pos = []
        neg = []
        for d in data:
            if d[1] in pos_label:
                # pos.append(d[0] if d[0].shape[0] == 1 else norm(d[0]))
                pos.append(d[0] if not norm_flag else norm(d[0]))
            elif neg_label is None or d[1] in neg_label:
                neg.append(d[0] if not norm_flag else norm(d[0]))
                # neg.append(d[0] if d[0].shape[0] == 1 else norm(d[0]))
        return pos, neg

    if neg_label is not None:
        pos, unl_neg = pos_neg_split(dataset, neg_label)
    else:
        pos, unl_neg = pos_neg_split(dataset)

    if not train:
        c = 0.0

    lab_pos, unl_pos = label_data(pos, c)

    if alpha:
        cur_alpha = len(unl_pos) / (len(unl_neg) + len(unl_pos))
        if cur_alpha > alpha:
            new_pos = int(alpha / (1 - alpha) * len(unl_neg))
            random.shuffle(unl_pos)
            unl_pos = unl_pos[:new_pos]
        else:
            new_neg = int((1 - alpha) / alpha * len(unl_pos))
            random.shuffle(unl_neg)
            unl_neg = unl_neg[:new_neg]

    prior = len(unl_pos) / (len(unl_pos) + len(unl_neg))
    # test_data = Dataset(test_pos + test_neg,
    #                     np.hstack((np.ones(len(test_pos)), np.zeros(len(test_neg)))),
    #                     np.zeros(len(test_pos) + len(test_neg)))

    data = Dataset(lab_pos + unl_pos + unl_neg,
                   np.concatenate((np.ones(len(lab_pos) + len(unl_pos)), np.zeros(len(unl_neg)))),
                   np.concatenate((np.ones(len(lab_pos)), np.zeros(len(unl_pos) + len(unl_neg)))))

    return data, prior


def make_shift_data(dataset_name="MNIST",
                    norm_flag=False,
                    **kwargs):
    """
        :param norm_flag:
        :param dataset_name: name of dataset: MNIST, CIFAR10
    """

    if norm_flag:
        trans = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    else:
        trans = transforms.ToTensor()

    if dataset_name == "MNIST":
        dataset = torchvision.datasets.MNIST("./data", train=False,
                                             transform=trans,
                                             download=True)
    elif dataset_name == "CIFAR10":
        dataset = torchvision.datasets.CIFAR10("./data", train=True,
                                               transform=trans,
                                               download=True)
    else:
        raise ValueError("Not supported dataset. Choose from MNIST, CIFAR10")
    data = []
    y = []
    for d in dataset:
        data.append(d[0])
        y.append(d[1])

    data = Dataset(data,
                   y,
                   np.zeros(len(data)))

    return data, 0.


def make_dataset(dataset_name,
                 train,
                 pos_label,
                 alpha,
                 neg_label,
                 return_prior,
                 need_ae,
                 encoder,
                 datamaker=make_data,
                 norm_flag=False,
                 need_svm_label=False,
                 c=0.5):
    data, pi = datamaker(dataset_name=dataset_name,
                         train=train,
                         pos_label=pos_label,
                         alpha=alpha,
                         neg_label=neg_label,
                         return_prior=return_prior,
                         norm_flag=norm_flag,
                         c=c)

    if need_ae:
        encoded_data = encode_dataset(encoder, data)
    else:
        encoded_data = data
        if need_svm_label:
            encoded_data.y = encoded_data.y * 2 - 1
            encoded_data.s = encoded_data.s * 2 - 1

    return encoded_data, pi


def label_data(dataset, c=0.5):
    lab = []
    unl = []
    for d in dataset:
        if np.random.rand() <= c:
            lab.append(d)
        else:
            unl.append(d)

    return lab, unl


def encode_dataset(encoder, dataset):
    new_s = dataset.s * 2 - 1
    new_y = dataset.y * 2 - 1

    imloader = torch.utils.data.DataLoader(dataset.data,
                                           batch_size=512)

    new_x = None
    for batch in imloader:
        batch_encoded = encoder(batch).detach().cpu()
        if new_x is None:
            new_x = batch_encoded
        else:
            new_x = torch.cat((new_x, batch_encoded), dim=0)

    return Dataset(new_x, new_y, new_s)
