import torch.optim as opt
import torch.utils.data
from sklearn import metrics

from classifiers import Net_CSI
from models.csi_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# code borrowed from https://github.com/alinlab/CSI

class CSI(nn.Module):
    def __init__(self, model=Net_CSI, simclr_dim=128, k_shift=4, lam=0.1, out_dim=128):
        super().__init__()
        # self.shift_trans = params['shift_trans']
        self.shift_trans = Rotation()
        # self.K_shift = params['K_shift']
        self.k_shift = k_shift
        self.sim_lambda = lam
        self.simclr_aug = get_simclr_augmentation((32, 32, 3)).to(device)
        self.hflip = HorizontalFlipLayer().to(device)

        self.model = model(out_dim=out_dim, simclr_dim=simclr_dim, shift=k_shift)
        self.train_features = None
        self.lambda_cls = None
        self.lambda_con = None

    def decision_function(self, test_data):
        self.model.eval()
        y_pred = np.array([])
        y_true = np.array([])

        test_loader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=64)

        for (images, labels, _) in test_loader:
            s_con = 0
            s_cls = 0

            images = images.to(device)
            images = self.hflip(images)

            images = torch.cat([self.shift_trans(images, k) for k in range(self.k_shift)])

            out = self.model(images, penultimate=True, shift=True)

            chunks = out['penultimate'].chunk(self.k_shift)
            chunks_s = out['shift'].chunk(self.k_shift)

            for i in range(self.k_shift):
                s_con += self.lambda_con[i] * (chunks[i] @ self.train_features[i].T).max(dim=1)[0]
                s_cls += self.lambda_cls[i] * (chunks_s[i][:, i])

            s_csi = s_con + s_cls
            y_pred = np.hstack((y_pred, s_csi.cpu().detach().numpy()))
            y_true = np.hstack((y_true, labels.cpu().detach().numpy()))

        return y_pred, y_true

    def update_decision_function(self, train_data):
        self.eval()

        dataloader = torch.utils.data.DataLoader(train_data.lab_data(lab=1),
                                                 batch_size=512)

        self.train_features = [[] for k in range(self.k_shift)]
        self.lambda_cls = torch.zeros(self.k_shift, device=device)
        self.lambda_con = torch.zeros(self.k_shift, device=device)

        for (images, _, labels) in dataloader:
            images = images.to(device)
            images = self.hflip(images)

            images = torch.cat([self.shift_trans(images, k) for k in range(self.k_shift)])

            out = self.model(images, penultimate=True, shift=True)

            chunks = out['penultimate'].chunk(self.k_shift)
            chunks_s = out['shift'].chunk(self.k_shift)

            for i in range(self.k_shift):
                self.train_features[i].append(chunks[i].detach())
                self.lambda_con[i] += torch.norm(chunks[i], p=2, dim=1).sum().detach()
                self.lambda_cls[i] += chunks_s[i].sum(dim=0)[i].detach()

        self.lambda_cls = len(train_data) / self.lambda_cls
        self.lambda_con = len(train_data) / self.lambda_con
        self.train_features = [torch.cat(f) for f in self.train_features]
        for i in range(self.k_shift):
            self.train_features[i] = self.train_features[i] / torch.norm(self.train_features[i], dim=1).view(-1, 1)

    def fit(self,
            train_data,
            num_epochs=60,
            lr=1e-3,
            gamma=0.99,
            batch_size=64,
            temp=0.5,
            verbose=False,
            test_data=None):

        # self.debug = dict()

        self.model.to(device)

        # self.debug['shift_loss'] = []
        # self.debug['shift_labels'] = []

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = opt.Adam(self.model.parameters(), lr=lr)
        scheduler = opt.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        # linear_optimizer = opt.Adam(self.model.linear.parameters(), lr=lr)

        data_loader = torch.utils.data.DataLoader(train_data.lab_data(lab=1),
                                                  batch_size=batch_size,
                                                  shuffle=True)
        for epoch in range(num_epochs):
            running_loss = 0
            for (images, _, labels) in data_loader:
                self.model.train()

                ### SimCLR loss ###
                images = images.to(device)
                images1, images2 = self.hflip(images.repeat(2, 1, 1, 1)).chunk(2)  # hflip

                labels = labels.to(device)

                images1 = torch.cat([self.shift_trans(images1, k) for k in range(self.k_shift)])
                images2 = torch.cat([self.shift_trans(images2, k) for k in range(self.k_shift)])

                shift_labels = torch.cat([torch.ones_like(labels) * k for k in range(self.k_shift)], 0)  # B -> 4B
                shift_labels = shift_labels.repeat(2)

                images_pair = torch.cat([images1, images2], dim=0)  # 8B
                # print(images_pair.shape)

                images_pair = self.simclr_aug(images_pair)  # transform

                # print(images_pair.shape)
                outputs_aux = self.model(images_pair, simclr=True, penultimate=True, shift=True)

                simclr = normalize(outputs_aux['simclr'])  # normalize
                sim_matrix = get_similarity_matrix(simclr, multi_gpu=False)
                loss_sim = NT_xent(sim_matrix, temperature=temp) * self.sim_lambda

                # self.debug['shift_loss'].append(outputs_aux['shift'].float())
                # self.debug['shift_labels'].append(shift_labels)
                loss_shift = criterion(outputs_aux['shift'].float(), shift_labels.long())

                ### total loss ###
                loss = loss_sim + loss_shift

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            scheduler.step()
            if test_data is not None and verbose:
                self.model.eval()
                self.update_decision_function(train_data)
                y_pred, y_true = self.decision_function(test_data)

                print_line = f"[{epoch:4}/{num_epochs:4}]: loss={running_loss: .4f}, test auc={metrics.roc_auc_score(y_true, y_pred):.4f}"

                print(print_line, flush=True)

        self.update_decision_function(train_data)


class PU_CSI(nn.Module):
    def __init__(self, model=Net_CSI, simclr_dim=128, k_shift=4, lam=0.1, out_dim=128, flag=True):
        """
        :param flag: if true other positive examples considered in contrastive loss as negatives, if false only
        unlabeled data considered as negative
        """
        super().__init__()
        self.k_shift = k_shift
        self.sim_lambda = lam
        self.simclr_aug = get_simclr_augmentation((32, 32, 3)).to(device)
        self.hflip = HorizontalFlipLayer().to(device)
        self.flag = flag

        self.model = model(out_dim=out_dim, simclr_dim=simclr_dim, shift=k_shift)
        self.train_features = {}
        self.lambda_cls = {}
        self.lambda_con = {}

    def decision_function(self, test_data):
        y_preds = {'csi': np.array([]), 'diff': np.array([]), 'combo': np.array([])}
        y_true = np.array([])

        test_loader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=64)

        for (images, labels, _) in test_loader:
            s_con_pos = 0
            s_con_unl = 0
            s_cls = 0

            images = images.to(device)
            images = self.hflip(images)

            images = torch.cat([torch.rot90(images, rot, (2, 3)) for rot in range(4)])

            out = self.model(images, penultimate=True, shift=True)

            chunks = out['penultimate'].chunk(4)
            chunks_s = out['shift'].chunk(4)

            for i in range(self.k_shift):
                s_con_pos += self.lambda_con['pos'][i] * (chunks[i] @ self.train_features['pos'][i].T).max(dim=1)[0]
                # s_con_unl += self.lambda_con['unl'][i] * (chunks[i] @ self.train_features['unl'][i].T).max(dim=1)[0]
                s_cls += self.lambda_cls['pos'][i] * (chunks_s[i][:, i])

            s_csi = s_con_pos + s_cls
            # s_diff = s_con_pos - s_con_unl
            # s_combo = s_con_pos + s_cls - s_con_unl

            y_preds['csi'] = np.hstack((y_preds['csi'], s_csi.cpu().detach().numpy()))
            # y_preds['diff'] = np.hstack((y_preds['diff'], s_diff.cpu().detach().numpy()))
            # y_preds['combo'] = np.hstack((y_preds['combo'], s_combo.cpu().detach().numpy()))

            y_true = np.hstack((y_true, labels.cpu().detach().numpy()))

        return y_preds['csi'], y_true

    def image_trans(self, images):
        images1, images2 = self.hflip(images.repeat(2, 1, 1, 1)).chunk(2)  # hflip
        images1 = torch.cat([torch.rot90(images1, rot, (2, 3)) for rot in range(4)])
        images2 = torch.cat([torch.rot90(images2, rot, (2, 3)) for rot in range(4)])
        images_pair = torch.cat([images1, images2], dim=0)  # 8B
        rot_labels = torch.cat([torch.ones(len(images)) * i for i in range(4)], dim=0)

        return images_pair, rot_labels.long().repeat(2)

    def fit(self,
            train_data,
            num_epochs=60,
            lr=1e-3,
            gamma=0.99,
            batch_size=64,
            batch_size_lab=64,
            batch_size_unl=16,
            temp=0.5,
            w=0.1,
            test_data=None,
            verbose=False):

        # self.debug = dict()

        self.model.to(device)

        # self.debug['shift_loss'] = []
        # self.debug['shift_labels'] = []

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = opt.Adam(self.model.parameters(), lr=lr)
        scheduler = opt.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        # data_loader = torch.utils.data.DataLoader(train_data,
        #                                           batch_size=batch_size,
        #                                           shuffle=True)

        lab_loader = torch.utils.data.DataLoader(train_data.lab_data(lab=1),
                                                 batch_size=batch_size_lab,
                                                 shuffle=True)

        unl_loader = torch.utils.data.DataLoader(train_data.lab_data(lab=0),
                                                 batch_size=batch_size_unl,
                                                 shuffle=True)

        for epoch in range(num_epochs):
            running_loss = 0
            # for (images, _, labels) in data_loader:
            for batch in zip(lab_loader, unl_loader):

                images_p, images_u = batch[0][0], batch[1][0]
                labels_p, labels_u = batch[0][2], batch[1][2]

                # if len(images_p) != len(images_u):
                #     continue

                # images_p, images_u = images[labels == 1], images[labels == 0]
                # labels_p, labels_u = labels[labels == 1], labels[labels == 0]

                # if images_p.shape[0] == 0:
                #     continue

                images_p, rot_labels_p = self.image_trans(images_p)
                labels_p = labels_p.repeat(4)

                if images_u.shape[0] > 0:
                    images_u, rot_labels_u = self.image_trans(images_u)
                    labels_u = labels_u.repeat(4)

                if images_u.shape[0] > 0:
                    images = torch.cat((images_p, images_u))
                    labels = torch.cat((labels_p, labels_u))
                else:
                    images = images_p
                    labels = labels_p

                images = images.to(device)
                labels = labels.to(device)
                rot_labels = rot_labels_p.to(device)

                self.model.train()

                ### SimCLR loss ###
                batch_size = images.size(0)

                images_pair = self.simclr_aug(images)  # simclr augment
                outputs_aux = self.model(images_pair, simclr=True, penultimate=True, shift=True)

                simclr = normalize(outputs_aux['simclr'])  # normalize
                sim_matrix = get_similarity_matrix(simclr)
                loss_csi = Semisupervised_NT_xent(sim_matrix, len(labels_p), len(labels_u), temperature=temp,
                                                  flag=self.flag, w=w) * self.sim_lambda

                loss_shift = criterion(outputs_aux['shift'].float()[:len(labels_p) * 2], rot_labels)

                loss = loss_csi + loss_shift

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            scheduler.step()
            if test_data is not None and verbose:
                self.model.eval()
                self.update_decision_function(train_data, 'pos')
                # self.update_decision_function(train_data, 'unl')

                y_pred, y_true = self.decision_function(test_data)

                auc_csi = metrics.roc_auc_score(y_true, y_pred)
                # auc_diff = metrics.roc_auc_score(y_true, y_preds['diff'])
                # auc_combo = metrics.roc_auc_score(y_true, y_preds['combo'])

                print_line = f"[{epoch:4}/{num_epochs:4}]: loss={running_loss: .4f}, auc_csi={auc_csi:.4f}"

                print(print_line)
        self.update_decision_function(train_data, 'pos')

    def update_decision_function(self, train_data, mode='pos'):
        self.eval()

        if mode == 'pos':
            label = 1
        elif mode == 'unl':
            label = 0
        else:
            raise ValueError(f'wrong mode {mode}')

        dataloader = torch.utils.data.DataLoader(train_data.lab_data(lab=label),
                                                 batch_size=512)

        self.train_features[mode] = [[] for k in range(self.k_shift)]
        self.lambda_cls[mode] = torch.zeros(self.k_shift, device=device)
        self.lambda_con[mode] = torch.zeros(self.k_shift, device=device)

        for (images, _, labels) in dataloader:
            batch_size = images.size(0)
            images = images.to(device)
            images = self.hflip(images)

            images = torch.cat([torch.rot90(images, rot, (2, 3)) for rot in range(4)])

            out = self.model(images, penultimate=True, shift=True)

            chunks = out['penultimate'].chunk(4)
            chunks_s = out['shift'].chunk(4)

            for i in range(4):
                self.train_features[mode][i].append(chunks[i].detach())
                self.lambda_con[mode][i] += torch.norm(chunks[i], p=2, dim=1).sum().detach()
                self.lambda_cls[mode][i] += chunks_s[i].sum(dim=0)[i].detach()

        self.lambda_cls[mode] = len(train_data) / self.lambda_cls[mode]
        self.lambda_con[mode] = len(train_data) / self.lambda_con[mode]
        self.train_features[mode] = [torch.cat(f) for f in self.train_features[mode]]

        for i in range(4):
            self.train_features[mode][i] = self.train_features[mode][i] / torch.norm(self.train_features[mode][i],
                                                                                     dim=1).view(-1, 1)
