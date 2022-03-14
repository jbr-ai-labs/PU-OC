import torch
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data

from models.base_models import OCModel, PUModelRandomBatch
from models.classifiers import Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# code DROCC is borrowed from https://github.com/microsoft/EdgeML


class DROCC(OCModel):
    def __init__(self,
                 model=Net,
                 lam=0.5,
                 radius=8,
                 gamma=2,
                 warmup_epochs=6,
                 ascent_step_size=0.001,
                 ascent_num_steps=50,
                 half=True):
        super().__init__(model, 0)
        self.lam = lam
        self.radius = radius
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.ascent_step_size = ascent_step_size
        self.ascent_num_steps = ascent_num_steps
        self.half = half

    def batch_loss(self, batch):
        data, target = batch[0], batch[2]
        data, target = data.to(device), target.to(device)

        # Data Processing
        data = data.to(torch.float)
        target = target.to(torch.float)
        target = torch.squeeze(target)

        # Extract the logits for cross entropy loss
        logits_start = self.model.forward_start(data)
        logits = self.model.forward_end(logits_start)

        logits = torch.squeeze(logits, dim=1)
        ce_loss = F.binary_cross_entropy_with_logits(logits, target)
        # Add to the epoch variable for printing average CE Loss

        '''
        Adversarial Loss is calculated only for the positive data points (label==1).
        '''
        if self.epoch >= self.warmup_epochs:
            logits_start = logits_start[target == 1]
            # AdvLoss
            if not self.half:
                adv_loss = self.one_class_adv_loss(data[target == 1].detach(), self.half)
            else:
                adv_loss = self.one_class_adv_loss(logits_start.detach(), self.half)

            loss = ce_loss + adv_loss * self.lam
        else:
            # If only CE based training has to be done
            loss = ce_loss

        return loss

    def one_class_adv_loss(self, x_train_data, half=True):
        """Computes the adversarial loss:
        1) Sample points initially at random around the positive training
            data points
        2) Gradient ascent to find the most optimal point in set N_i(r)
            classified as +ve (label=0). This is done by maximizing
            the CE loss wrt label 0
        3) Project the points between spheres of radius R and gamma * R
            (set N_i(r))
        4) Pass the calculated adversarial points through the model,
            and calculate the CE loss wrt target class 0

        Parameters
        ----------
        x_train_data: Batch of data to compute loss on.
        """
        batch_size = len(x_train_data)
        # Randomly sample points around the training data
        # We will perform SGD on these to find the adversarial points
        x_adv = torch.randn(x_train_data.shape).to(device).detach().requires_grad_()
        x_adv_sampled = x_adv + x_train_data

        for step in range(self.ascent_num_steps):
            with torch.enable_grad():

                new_targets = torch.zeros(batch_size, 1).to(device)
                # new_targets = (1 - targets).to(self.device)
                new_targets = torch.squeeze(new_targets)
                new_targets = new_targets.to(torch.float)

                if half:
                    logits = self.model.forward_end(x_adv_sampled)
                else:
                    logits = self.model(x_adv_sampled)

                logits = torch.squeeze(logits, dim=1)
                new_loss = F.binary_cross_entropy_with_logits(logits, new_targets)

                grad = torch.autograd.grad(new_loss, [x_adv_sampled])[0]
                grad_norm = torch.norm(grad, p=2, dim=tuple(range(1, grad.dim())))
                grad_norm = grad_norm.view(-1, *[1] * (grad.dim() - 1))
                grad_normalized = grad / grad_norm
            with torch.no_grad():
                x_adv_sampled.add_(self.ascent_step_size * grad_normalized)

            if (step + 1) % 10 == 0:
                # Project the normal points to the set N_i(r)
                h = x_adv_sampled - x_train_data
                norm_h = torch.sqrt(torch.sum(h ** 2,
                                              dim=tuple(range(1, h.dim()))))
                alpha = torch.clamp(norm_h, self.radius,
                                    self.gamma * self.radius).to(device)
                # Make use of broadcast to project h
                proj = (alpha / norm_h).view(-1, *[1] * (h.dim() - 1))
                h = proj * h
                x_adv_sampled = x_train_data + h  # These adv_points are now on the surface of hyper-sphere

        if half:
            adv_pred = self.model.forward_end(x_adv_sampled)
        else:
            adv_pred = self.model(x_adv_sampled)

        adv_pred = torch.squeeze(adv_pred, dim=1)
        adv_loss = F.binary_cross_entropy_with_logits(adv_pred, new_targets)

        return adv_loss


# class DROCC(nn.Module):
#     def __init__(self, ):
#         super().__init__()
#
#         self.model = CIFAR10_LeNet()
#
#     def run_train(self,
#                   train_data,
#                   test_data,
#                   lamda=0.5,
#                   radius=8,
#                   gamma=2,
#                   verbose=False,
#                   learning_rate=1e-3,
#                   total_epochs=30,
#                   only_ce_epochs=6,
#                   ascent_step_size=0.001,
#                   ascent_num_steps=50,
#                   gamma_lr=1,
#                   batch_size=128,
#                   half=True):
#
#         self.best_score = -np.inf
#         best_model = None
#         self.ascent_num_steps = ascent_num_steps
#         self.ascent_step_size = ascent_step_size
#         self.lamda = lamda
#         self.radius = radius
#         self.gamma = gamma
#
#         self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
#         lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma_lr)
#
#         train_loader = torch.utils.data.DataLoader(train_data,
#                                                    batch_size=batch_size,
#                                                    shuffle=True)
#
#         test_loader = torch.utils.data.DataLoader(test_data,
#                                                   batch_size=batch_size,
#                                                   shuffle=True)
#
#         for epoch in range(total_epochs):
#             # Make the weights trainable
#             self.model.train()
#
#             # Placeholder for the respective 2 loss values
#             epoch_adv_loss = torch.tensor([0]).type(torch.float32).to(device)  # AdvLoss
#             epoch_ce_loss = 0  # Cross entropy Loss
#
#             batch_idx = -1
#             for data, target, _ in train_loader:
#                 batch_idx += 1
#                 data, target = data.to(device), target.to(device)
#                 # Data Processing
#                 data = data.to(torch.float)
#                 target = target.to(torch.float)
#                 target = torch.squeeze(target)
#
#                 self.optimizer.zero_grad()
#
#                 # Extract the logits for cross entropy loss
#                 logits_start = self.model.half_forward_start(data)
#                 logits = self.model.half_forward_end(logits_start)
#
#                 logits = torch.squeeze(logits, dim=1)
#                 ce_loss = F.binary_cross_entropy_with_logits(logits, target)
#                 # Add to the epoch variable for printing average CE Loss
#                 epoch_ce_loss += ce_loss
#
#                 '''
#                 Adversarial Loss is calculated only for the positive data points (label==1).
#                 '''
#                 if epoch >= only_ce_epochs:
#                     logits_start = logits_start[target == 1]
#                     # AdvLoss
#                     if not half:
#                         adv_loss = self.one_class_adv_loss(data[target == 1].detach(), target[target == 1], half)
#                     else:
#                         adv_loss = self.one_class_adv_loss(logits_start.detach(), target[target == 1], half)
#                     epoch_adv_loss += adv_loss
#
#                     loss = ce_loss + adv_loss * self.lamda
#                 else:
#                     # If only CE based training has to be done
#                     loss = ce_loss
#
#                 # Backprop
#                 loss.backward()
#                 self.optimizer.step()
#
#             epoch_ce_loss = epoch_ce_loss / (batch_idx + 1)  # Average CE Loss
#             epoch_adv_loss = epoch_adv_loss / (batch_idx + 1)  # Average AdvLoss
#
#             if verbose:
#                 test_score = self.test(test_loader)
#                 if test_score > self.best_score:
#                     self.best_score = test_score
#                     best_model = copy.deepcopy(self.model)
#
#                 print('Epoch: {}, CE Loss: {}, AdvLoss: {}, {}: {}'.format(
#                     epoch, epoch_ce_loss.item(), epoch_adv_loss.item(),
#                     'AUC', test_score))
#             lr_scheduler.step()
#         if verbose:
#             self.model = copy.deepcopy(best_model)
#             print('\nBest test {}: {}'.format(
#                 'AUC', self.best_score
#             ))
#
#     def test(self, test_loader, metric='AUC'):
#         """Evaluate the model on the given test dataset.
#         Parameters
#         ----------
#         test_loader: Dataloader object for the test dataset.
#         metric: Metric used for evaluation (AUC / F1).
#         """
#         self.model.eval()
#         label_score = []
#         batch_idx = -1
#         for data, target, _ in test_loader:
#             batch_idx += 1
#             data, target = data.to(device), target.to(device)
#             data = data.to(torch.float)
#             target = target.to(torch.float)
#             target = torch.squeeze(target)
#
#             logits = self.model(data)
#             logits = torch.squeeze(logits, dim=1)
#             sigmoid_logits = torch.sigmoid(logits)
#             scores = logits
#             label_score += list(zip(target.cpu().data.numpy().tolist(),
#                                     scores.cpu().data.numpy().tolist()))
#         # Compute test score
#         labels, scores = zip(*label_score)
#         labels = np.array(labels)
#         scores = np.array(scores)
#         if metric == 'AUC':
#             test_metric = roc_auc_score(labels, scores)
#         if metric == 'alpha':
#             test_metric = (scores > 0.5).mean()
#         return test_metric
#
#     def one_class_adv_loss(self, x_train_data, targets, half=True):
#         """Computes the adversarial loss:
#         1) Sample points initially at random around the positive training
#             data points
#         2) Gradient ascent to find the most optimal point in set N_i(r)
#             classified as +ve (label=0). This is done by maximizing
#             the CE loss wrt label 0
#         3) Project the points between spheres of radius R and gamma * R
#             (set N_i(r))
#         4) Pass the calculated adversarial points through the model,
#             and calculate the CE loss wrt target class 0
#
#         Parameters
#         ----------
#         x_train_data: Batch of data to compute loss on.
#         """
#         batch_size = len(x_train_data)
#         # Randomly sample points around the training data
#         # We will perform SGD on these to find the adversarial points
#         x_adv = torch.randn(x_train_data.shape).to(device).detach().requires_grad_()
#         x_adv_sampled = x_adv + x_train_data
#
#         for step in range(self.ascent_num_steps):
#             with torch.enable_grad():
#
#                 new_targets = torch.zeros(batch_size, 1).to(device)
#                 # new_targets = (1 - targets).to(self.device)
#                 new_targets = torch.squeeze(new_targets)
#                 new_targets = new_targets.to(torch.float)
#
#                 if half:
#                     logits = self.model.half_forward_end(x_adv_sampled)
#                 else:
#                     logits = self.model(x_adv_sampled)
#
#                 logits = torch.squeeze(logits, dim=1)
#                 new_loss = F.binary_cross_entropy_with_logits(logits, new_targets)
#
#                 grad = torch.autograd.grad(new_loss, [x_adv_sampled])[0]
#                 grad_norm = torch.norm(grad, p=2, dim=tuple(range(1, grad.dim())))
#                 grad_norm = grad_norm.view(-1, *[1] * (grad.dim() - 1))
#                 grad_normalized = grad / grad_norm
#             with torch.no_grad():
#                 x_adv_sampled.add_(self.ascent_step_size * grad_normalized)
#
#             if (step + 1) % 10 == 0:
#                 # Project the normal points to the set N_i(r)
#                 h = x_adv_sampled - x_train_data
#                 norm_h = torch.sqrt(torch.sum(h ** 2,
#                                               dim=tuple(range(1, h.dim()))))
#                 alpha = torch.clamp(norm_h, self.radius,
#                                     self.gamma * self.radius).to(device)
#                 # Make use of broadcast to project h
#                 proj = (alpha / norm_h).view(-1, *[1] * (h.dim() - 1))
#                 h = proj * h
#                 x_adv_sampled = x_train_data + h  # These adv_points are now on the surface of hyper-sphere
#
#         if half:
#             adv_pred = self.model.half_forward_end(x_adv_sampled)
#         else:
#             adv_pred = self.model(x_adv_sampled)
#
#         adv_pred = torch.squeeze(adv_pred, dim=1)
#         adv_loss = F.binary_cross_entropy_with_logits(adv_pred, (new_targets))
#
#         return adv_loss
#
#     def save(self, path):
#         torch.save(self.model.state_dict(), os.path.join(path, 'model.pt'))
#
#     def load(self, path):
#         self.model.load_state_dict(torch.load(os.path.join(path, 'model.pt')))

class PU_DROCC(PUModelRandomBatch):
    def __init__(self,
                 model=Net,
                 lam=0.5,
                 radius=8,
                 gamma=2,
                 warmup_epochs=6,
                 ascent_step_size=0.001,
                 ascent_num_steps=50,
                 half=True):
        super().__init__(model, 0)
        self.lam = lam
        self.radius = radius
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.ascent_step_size = ascent_step_size
        self.ascent_num_steps = ascent_num_steps
        self.half = half

    def batch_loss(self, batch):
        data, target = batch[0], batch[2]
        data, target = data.to(device), target.to(device)

        lab_ind = target == 1
        unl_ind = target == 0

        # lab_cnt = max(lab_ind.sum(), 1)
        unl_cnt = max(unl_ind.sum(), 1)

        # Extract the logits for cross entropy loss
        logits_start = self.model.forward_start(data)
        logits = self.model.forward_end(logits_start[lab_ind])

        logits = torch.squeeze(logits, dim=1)
        ce_loss = F.binary_cross_entropy_with_logits(logits, target[lab_ind])
        # Add to the epoch variable for printing average CE Loss

        '''
        Adversarial Loss is calculated only for the positive data points (label==1).
        '''
        if self.epoch >= self.warmup_epochs and unl_cnt > 1:
            logits_start = logits_start[unl_ind]
            # AdvLoss
            if not self.half:
                adv_loss = self.one_class_adv_loss(data[unl_ind].detach(), self.half)
            else:
                adv_loss = self.one_class_adv_loss(logits_start[unl_ind].detach(), self.half)

            loss = ce_loss + adv_loss * self.lam
        else:
            # If only CE based training has to be done
            loss = ce_loss
        return loss

    def one_class_adv_loss(self, x_train_data, half=True):
        """Computes the adversarial loss:
        1) Sample points initially at random around the positive training
            data points
        2) Gradient ascent to find the most optimal point in set N_i(r)
            classified as +ve (label=0). This is done by maximizing
            the CE loss wrt label 0
        3) Project the points between spheres of radius R and gamma * R
            (set N_i(r))
        4) Pass the calculated adversarial points through the model,
            and calculate the CE loss wrt target class 0

        Parameters
        ----------
        x_train_data: Batch of data to compute loss on.
        """
        batch_size = len(x_train_data)
        # Randomly sample points around the training data
        # We will perform SGD on these to find the adversarial points
        x_adv = torch.randn(x_train_data.shape).to(device).detach().requires_grad_()
        x_adv_sampled = x_adv + x_train_data

        for step in range(self.ascent_num_steps):
            with torch.enable_grad():

                new_targets = torch.zeros(batch_size, 1).to(device)
                # new_targets = (1 - targets).to(self.device)
                new_targets = torch.squeeze(new_targets)
                new_targets = new_targets.to(torch.float)

                if half:
                    logits = self.model.forward_end(x_adv_sampled)
                else:
                    logits = self.model(x_adv_sampled)

                logits = torch.squeeze(logits, dim=1)
                new_loss = F.binary_cross_entropy_with_logits(logits, new_targets)

                grad = torch.autograd.grad(new_loss, [x_adv_sampled])[0]
                grad_norm = torch.norm(grad, p=2, dim=tuple(range(1, grad.dim())))
                grad_norm = grad_norm.view(-1, *[1] * (grad.dim() - 1))
                grad_normalized = grad / grad_norm
            with torch.no_grad():
                x_adv_sampled.add_(self.ascent_step_size * grad_normalized)

            if (step + 1) % 10 == 0:
                # Project the normal points to the set N_i(r)
                h = x_adv_sampled - x_train_data
                norm_h = torch.sqrt(torch.sum(h ** 2,
                                              dim=tuple(range(1, h.dim()))))
                alpha = torch.clamp(norm_h, self.radius,
                                    self.gamma * self.radius).to(device)
                # Make use of broadcast to project h
                proj = (alpha / norm_h).view(-1, *[1] * (h.dim() - 1))
                h = proj * h
                x_adv_sampled = x_train_data + h  # These adv_points are now on the surface of hyper-sphere

        if half:
            adv_pred = self.model.forward_end(x_adv_sampled)
        else:
            adv_pred = self.model(x_adv_sampled)

        adv_pred = torch.squeeze(adv_pred, dim=1)
        adv_loss = F.binary_cross_entropy_with_logits(adv_pred, new_targets)

        return adv_loss

# class PU_DROCC(nn.Module):
#     def __init__(self, ):
#         super().__init__()
#
#         self.model = CIFAR10_LeNet()
#
#     def run_train(self,
#                   train_data,
#                   test_data,
#                   lamda=0.5,
#                   radius=1,
#                   gamma=2,
#                   verbose=False,
#                   learning_rate=5e-4,
#                   total_epochs=20,
#                   only_ce_epochs=2,
#                   ascent_step_size=5e-6,
#                   ascent_num_steps=10,
#                   gamma_lr=0.96,
#                   batch_size=512,
#                   half=True):
#
#         self.best_score = -np.inf
#         best_model = None
#         self.ascent_num_steps = ascent_num_steps
#         self.ascent_step_size = ascent_step_size
#         self.lamda = lamda
#         self.radius = radius
#         self.gamma = gamma
#
#         self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
#         lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma_lr)
#
#         train_loader = torch.utils.data.DataLoader(train_data,
#                                                    batch_size=batch_size,
#                                                    shuffle=True)
#
#         test_loader = torch.utils.data.DataLoader(test_data,
#                                                   batch_size=batch_size,
#                                                   shuffle=True)
#
#         for epoch in range(total_epochs):
#             # Make the weights trainable
#             self.model.train()
#
#             # Placeholder for the respective 2 loss values
#             epoch_adv_loss = torch.tensor([0]).type(torch.float32).to(device)  # AdvLoss
#             epoch_ce_loss = 0  # Cross entropy Loss
#
#             batch_idx = -1
#             for data, _, target in train_loader:
#                 batch_idx += 1
#                 data, target = data.to(device), target.to(device)
#                 # Data Processing
#                 data = data.to(torch.float)
#                 target = target.to(torch.float)
#                 target = torch.squeeze(target)
#
#                 self.optimizer.zero_grad()
#
#                 lab_ind = target == 1
#                 unl_ind = target == 0
#
#                 # lab_cnt = max(lab_ind.sum(), 1)
#                 unl_cnt = max(unl_ind.sum(), 1)
#
#                 # Extract the logits for cross entropy loss
#                 logits_start = self.model.half_forward_start(data)
#                 logits = self.model.half_forward_end(logits_start[lab_ind])
#
#                 logits = torch.squeeze(logits, dim=1)
#                 ce_loss = F.binary_cross_entropy_with_logits(logits, target[lab_ind])
#                 # Add to the epoch variable for printing average CE Loss
#                 epoch_ce_loss += ce_loss
#
#                 '''
#                 Adversarial Loss is calculated only for the positive data points (label==1).
#                 '''
#                 if epoch >= only_ce_epochs and unl_cnt > 1:
#                     logits_start = logits_start[unl_ind]
#                     # AdvLoss
#                     if not half:
#                         adv_loss = self.one_class_adv_loss(data[unl_ind].detach(), target[unl_ind], half)
#                     else:
#                         adv_loss = self.one_class_adv_loss(logits_start.detach(), target[unl_ind], half)
#                     epoch_adv_loss += adv_loss
#
#                     loss = ce_loss + adv_loss * self.lamda
#                 else:
#                     # If only CE based training has to be done
#                     loss = ce_loss
#
#                 # Backprop
#                 loss.backward()
#                 self.optimizer.step()
#
#             epoch_ce_loss = epoch_ce_loss / (batch_idx + 1)  # Average CE Loss
#             epoch_adv_loss = epoch_adv_loss / (batch_idx + 1)  # Average AdvLoss
#
#             if verbose:
#                 test_score = self.test(test_loader)
#                 if test_score > self.best_score:
#                     self.best_score = test_score
#                     best_model = copy.deepcopy(self.model)
#
#                 print('Epoch: {}, CE Loss: {}, AdvLoss: {}, {}: {}'.format(
#                     epoch, epoch_ce_loss.item(), epoch_adv_loss.item(),
#                     'AUC', test_score))
#             lr_scheduler.step()
#         if verbose:
#             self.model = copy.deepcopy(best_model)
#             print('\nBest test {}: {}'.format(
#                 'AUC', self.best_score
#             ))
#
#     def test(self, test_loader, metric='AUC'):
#         """Evaluate the model on the given test dataset.
#         Parameters
#         ----------
#         test_loader: Dataloader object for the test dataset.
#         metric: Metric used for evaluation (AUC / F1).
#         """
#         self.model.eval()
#         label_score = []
#         batch_idx = -1
#         for data, target, _ in test_loader:
#             batch_idx += 1
#             data, target = data.to(device), target.to(device)
#             data = data.to(torch.float)
#             target = target.to(torch.float)
#             target = torch.squeeze(target)
#
#             logits = self.model(data)
#             logits = torch.squeeze(logits, dim=1)
#             sigmoid_logits = torch.sigmoid(logits)
#             scores = logits
#             label_score += list(zip(target.cpu().data.numpy().tolist(),
#                                     scores.cpu().data.numpy().tolist()))
#         # Compute test score
#         labels, scores = zip(*label_score)
#         labels = np.array(labels)
#         scores = np.array(scores)
#         if metric == 'AUC':
#             test_metric = roc_auc_score(labels, scores)
#         if metric == 'alpha':
#             test_metric = (scores > 0.5).mean()
#         return test_metric
#
#     def one_class_adv_loss(self, x_train_data, targets, half=True):
#         """Computes the adversarial loss:
#         1) Sample points initially at random around the positive training
#             data points
#         2) Gradient ascent to find the most optimal point in set N_i(r)
#             classified as +ve (label=0). This is done by maximizing
#             the CE loss wrt label 0
#         3) Project the points between spheres of radius R and gamma * R
#             (set N_i(r))
#         4) Pass the calculated adversarial points through the model,
#             and calculate the CE loss wrt target class 0
#
#         Parameters
#         ----------
#         x_train_data: Batch of data to compute loss on.
#         """
#         batch_size = len(x_train_data)
#         # Randomly sample points around the training data
#         # We will perform SGD on these to find the adversarial points
#         x_adv = torch.randn(x_train_data.shape).to(device).detach().requires_grad_()
#         x_adv_sampled = x_adv + x_train_data
#
#         for step in range(self.ascent_num_steps):
#             with torch.enable_grad():
#
#                 new_targets = torch.zeros(batch_size, 1).to(device)
#                 # new_targets = (1 - targets).to(self.device)
#                 new_targets = torch.squeeze(new_targets)
#                 new_targets = new_targets.to(torch.float)
#
#                 if half:
#                     logits = self.model.half_forward_end(x_adv_sampled)
#                 else:
#                     logits = self.model(x_adv_sampled)
#
#                 logits = torch.squeeze(logits, dim=1)
#                 new_loss = F.binary_cross_entropy_with_logits(logits, new_targets)
#
#                 grad = torch.autograd.grad(new_loss, [x_adv_sampled])[0]
#                 grad_norm = torch.norm(grad, p=2, dim=tuple(range(1, grad.dim())))
#                 grad_norm = grad_norm.view(-1, *[1] * (grad.dim() - 1))
#                 grad_normalized = grad / grad_norm
#             with torch.no_grad():
#                 x_adv_sampled.add_(self.ascent_step_size * grad_normalized)
#
#             if (step + 1) % 10 == 0:
#                 # Project the normal points to the set N_i(r)
#                 h = x_adv_sampled - x_train_data
#                 norm_h = torch.sqrt(torch.sum(h ** 2,
#                                               dim=tuple(range(1, h.dim()))))
#                 alpha = torch.clamp(norm_h, self.radius,
#                                     self.gamma * self.radius).to(device)
#                 # Make use of broadcast to project h
#                 proj = (alpha / norm_h).view(-1, *[1] * (h.dim() - 1))
#                 h = proj * h
#                 x_adv_sampled = x_train_data + h  # These adv_points are now on the surface of hyper-sphere
#
#         if half:
#             adv_pred = self.model.half_forward_end(x_adv_sampled)
#         else:
#             adv_pred = self.model(x_adv_sampled)
#
#         adv_pred = torch.squeeze(adv_pred, dim=1)
#         adv_loss = F.binary_cross_entropy_with_logits(adv_pred, (new_targets))
#
#         return adv_loss
#
#     def save(self, path):
#         torch.save(self.model.state_dict(), os.path.join(path, 'model.pt'))
#
#     def load(self, path):
#         self.model.load_state_dict(torch.load(os.path.join(path, 'model.pt')))
