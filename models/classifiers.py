import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision import models


# neural classifiers

class Net(nn.Module):
    def __init__(self, inp_dim=(32, 32, 3), out_dim=1, hid_dim_full=128, bayes=False, bn=False, prob=True):
        super(Net, self).__init__()
        self.bayes = bayes
        self.bn = bn
        self.prob = prob

        self.conv1 = nn.Conv2d(3, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(16, 32, 5, padding=2)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
        self.conv5 = nn.Conv2d(32, 32, 1)
        self.conv6 = nn.Conv2d(32, 4, 1)
        if bn:
            self.bn1 = nn.BatchNorm2d(16)
            self.bn2 = nn.BatchNorm2d(16)
            self.bn3 = nn.BatchNorm2d(32)
            self.bn4 = nn.BatchNorm2d(32)
            self.bn5 = nn.BatchNorm2d(32)
            self.bn6 = nn.BatchNorm2d(4)

        self.conv_to_fc = 8 * 8 * 4
        self.fc1 = nn.Linear(self.conv_to_fc, hid_dim_full)
        self.fc2 = nn.Linear(hid_dim_full, int(hid_dim_full // 2))
        if self.bayes:
            self.out_mean = nn.Linear(int(hid_dim_full // 2), out_dim)
            self.out_logvar = nn.Linear(int(hid_dim_full // 2), out_dim)
        else:
            self.out = nn.Linear(int(hid_dim_full // 2), out_dim)

    def forward(self, x, return_params=False, sample_noise=False):
        x = self.forward_start(x)
        return self.forward_end(x, return_params, sample_noise)

    def forward_start(self, x, return_params=False, sample_noise=False):
        if self.bn:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn3(self.conv4(x)))
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))

        return x

    def forward_end(self, x, return_params=False, sample_noise=False):
        if self.bn:
            x = F.relu(self.bn5(self.conv5(x)))
            x = F.relu(self.bn6(self.conv6(x)))
        else:
            x = F.relu(self.conv5(x))
            x = F.relu(self.conv6(x))

        x = x.view(-1, self.conv_to_fc)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        if self.bayes:
            mean, logvar = self.out_mean(x), self.out_logvar(x)
            var = torch.exp(logvar * .5)
            if sample_noise:
                x = mean + var * torch.randn_like(var)
            else:
                x = mean
        else:
            mean = self.out(x)
            var = torch.zeros_like(mean) + 1e-3
            x = mean

        if self.prob:
            p = torch.sigmoid(x)
        else:
            p = x

        if return_params:
            return p, mean, var
        else:
            return p


class Net_CSI(nn.Module):
    def __init__(self, out_dim=128, hid_dim_full=128, simclr_dim=128, num_classes=2, bn=False, shift=4):
        super(Net_CSI, self).__init__()
        self.bn = bn

        self.conv1 = nn.Conv2d(3, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(16, 32, 5, padding=2)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
        self.conv5 = nn.Conv2d(32, 32, 1)
        self.conv6 = nn.Conv2d(32, 4, 1)
        if bn:
            self.bn1 = nn.BatchNorm2d(16)
            self.bn2 = nn.BatchNorm2d(16)
            self.bn3 = nn.BatchNorm2d(32)
            self.bn4 = nn.BatchNorm2d(32)
            self.bn5 = nn.BatchNorm2d(32)
            self.bn6 = nn.BatchNorm2d(4)

        self.conv_to_fc = 8 * 8 * 4
        self.fc1 = nn.Linear(self.conv_to_fc, hid_dim_full)
        self.fc2 = nn.Linear(hid_dim_full, int(hid_dim_full // 2))

        self.features = nn.Linear(int(hid_dim_full // 2), out_dim)
        self.simclr_layer = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, simclr_dim),
        )

        self.shift_cls_layer = nn.Linear(out_dim, shift)

        self.linear = nn.Linear(out_dim, num_classes)
        self.joint_distribution_layer = nn.Linear(out_dim, shift * num_classes)

    def forward(self, x, penultimate=True, simclr=True, shift=True):
        if self.bn:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))
            x = F.relu(self.bn5(self.conv5(x)))
            x = F.relu(self.bn6(self.conv6(x)))
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            x = F.relu(self.conv5(x))
            x = F.relu(self.conv6(x))

        x = x.view(-1, self.conv_to_fc)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        features = self.features(x)

        res = dict()

        if penultimate:
            res['penultimate'] = features

        if simclr:
            res['simclr'] = self.simclr_layer(features)

        if shift:
            res['shift'] = self.shift_cls_layer(features)

        return res


class Net_CNN(nn.Module):
    def __init__(self, D=4096, pre_trained_flag=True):
        super().__init__()

        self.classifier = nn.Sequential(nn.ReLU(),
                                        nn.Linear(D, 1),
                                        nn.Sigmoid())
        self.D = D
        self.model = models.resnet18(pretrained=pre_trained_flag)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Linear(num_ftrs, D),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(D, D))

        self.inm = nn.InstanceNorm1d(1, affine=False)

    def forward_start(self, batch_x):
        if batch_x.shape[1] == 1:
            batch_x = batch_x.repeat(1, 3, 1, 1)

        out1 = self.model(batch_x)
        out1 = out1.view(int(batch_x.shape[0]), 1, self.D)
        out1 = self.inm(out1)
        out1 = out1.view(int(batch_x.shape[0]), self.D)

        return out1

    def forward_end(self, batch_x):
        return self.classifier(batch_x)


class Net_DROCC(nn.Module):
    """
    Old
    """

    def __init__(self):
        super().__init__()

        self.rep_dim = 128
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)
        self.fc2 = nn.Linear(self.rep_dim, int(self.rep_dim / 2), bias=False)
        self.fc3 = nn.Linear(int(self.rep_dim / 2), 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def half_forward_start(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        return x

    def half_forward_end(self, x):
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def name(self):
        return "lenet"

    def predict(self, input):
        return self.forward(input.float())
