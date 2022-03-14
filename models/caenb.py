import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# autoencoder for SVM-base models

class Encoder(nn.Module):
    def __init__(self, dim, dataset="Abnormal"):
        super(Encoder, self).__init__()

        if dataset == "MNIST":
            channels = 1
            self.mid = 392
        elif dataset == "CIFAR10":
            channels = 3
            self.mid = 512
        elif dataset == "Abnormal":
            channels = 3
            self.mid = 20000
        else:
            raise ValueError(f"Wrong dataset {dataset}")

        self.conv1 = nn.Conv2d(channels, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.dense1 = nn.Linear(self.mid, dim)

    def forward(self, img):
        x = img.to(device)
        x = self.conv1(x)
        x = F.elu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.elu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.elu(x)

        x = x.view(-1, self.mid)
        x = self.dense1(x)
        x = F.dropout(x, training=self.training)
        x = F.elu(x)

        return x


class Decoder(nn.Module):
    def __init__(self, dim, dataset="Abnormal"):
        super(Decoder, self).__init__()

        if dataset == "MNIST":
            channels = 1
            self.mid = 392
        elif dataset == "CIFAR10":
            channels = 3
            self.mid = 512
        elif dataset == "Abnormal":
            channels = 3
            self.mid = 20000
        else:
            raise ValueError(f"Wrong dataset {dataset}")
        self.deconv3 = nn.ConvTranspose2d(8, 8, 3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(8, 16, 3, stride=1, padding=1)
        self.deconv1 = nn.ConvTranspose2d(16, channels, 3, stride=1, padding=1)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dense1 = nn.Linear(dim, self.mid)


    def forward(self, encode):
        x = self.dense1(encode)
        x = F.dropout(x, training=self.training)
        x = F.elu(x)

        if self.mid == 392:
            x = x.view(x.size(0), 8, 7, 7)
        elif self.mid == 512:
            x = x.view(encode.shape[0], 8, 8, 8)
        elif self.mid == 20000:
            x = x.view(encode.shape[0], 8, 50, 50)

        x = self.deconv3(x)
        x = F.elu(x)

        x = self.upsample2(x)

        x = self.deconv2(x)
        x = F.elu(x)

        x = self.upsample1(x)

        x = self.deconv1(x)
        x = torch.sigmoid(x)

        return x


class CAE(nn.Module):
    def __init__(self, dim=128, dataset="Abnormal"):
        super(CAE, self).__init__()

        self.encoder = Encoder(dim, dataset=dataset)
        self.decoder = Decoder(dim, dataset=dataset)

    def forward(self, img):
        x = self.encoder(img)
        x = self.decoder(x)

        return x

    def run_train(self, dataset, lr=1e-3, num_epochs=100, batchsize=128):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        trainloader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batchsize,
                                                  shuffle=True,
                                                  num_workers=2)

        best_ae_wts = copy.deepcopy(self.state_dict())
        best_loss = np.inf

        for epoch in range(num_epochs):

            # Each epoch has a training and validation phase
            self.train()  # Set model to training mode

            running_loss = 0.0

            # Iterate over data.
            for (inputs, _, _) in trainloader:
                inputs = inputs.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                outputs = self(inputs)
                loss = F.mse_loss(inputs, outputs)

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss

            print(f'[{epoch:2}/{num_epochs}] Loss: {epoch_loss:.4f}')

            # deep copy the model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_ae_wts = copy.deepcopy(self.state_dict())
                self.load_state_dict(best_ae_wts)
                torch.save(self.state_dict(), "ae.pcl")

        print(f'[{epoch:2}/{num_epochs}] Best Loss: {best_loss:4f}')


def get_encoder(path, dataset, pref=True):
    if pref:
        path = "PU-vs-OC/AE weights/" + path

    ae = CAE(dataset=dataset)
    ae = ae.to(device)
    ae.load_state_dict(torch.load(path, map_location=device))
    ae.eval()
    encoder = ae.encoder
    return encoder
