# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 16:13:28 2019

@author: Arthur
Torch implementation of a similar form of NN as in Bolton et al
"""

import numpy as np
import mlflow
import os.path
from sklearn.preprocessing import normalize

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim
from torch.nn import Module
import torch.nn
import torch.nn.functional as F
from torchvision import models

import matplotlib.pyplot as plt
from enum import Enum

# Script parameters
batch_size = 8
learning_rate = 1e-3
nb_epochs = 200
print_loss_every = 20
data_location = '/data/ag7531/'
figures_directory = 'figures'

# Device selection
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Selected device:{}'.format(str(device)))

# Log the parameters with mlflow
mlflow.log_param('Batch size', batch_size)
mlflow.log_param('Learning rate', learning_rate)
mlflow.log_param('Device', device)


class NN_State(Enum):
    TRAIN = 0
    TEST = 1

class Dataset_psi_s(Dataset):
    """Loads the data from the disk into memory and produces data items on
    demand."""
    def __init__(self, data_location, file_name_psi, file_name_sx,
                 file_name_sy):
        # Load the data from the disk
        self.psi = np.load(os.path.join(data_location, file_name_psi))
        self.sx = np.load(os.path.join(data_location, file_name_sx))
        self.sy = np.load(os.path.join(data_location, file_name_sy))
        # TODO will have to remove this once read_data is run again
        self.psi = np.swapaxes(self.psi, 0, 2)
        self.sx = np.swapaxes(self.sx, 0, 2)
        self.sy = np.swapaxes(self.sy, 0, 2)
        assert self.psi.shape[0] == self.sx.shape[0] == self.sy.shape[0], \
            'Error: the lengths of the arrays differ'
        self.data_length = self.psi.shape[0]
        self._n_output_targets = 2* (self.sx.shape[1])**2
        self._pre_process()

    def _pre_process(self):
        """Operates some basic pre-processing on the data, such as mean
        removal and rescaling."""
        # TODO I should only use the mean and std of the training data really
        self.width, self.height = self.psi.shape[-2:]
        # Add the channel dimension to the input
        self.psi = self.psi.reshape(-1, 1, self.width, self.height)
        # Remove mean
        self.psi = (self.psi - np.mean(self.psi)) / np.std(self.psi) 
        self.sx = self.sx.reshape(self.sx.shape[0], -1)
        self.sy = self.sy.reshape(self.sy.shape[0], -1)
        self.std_sx = np.std(self.sx)
        self.std_sy = np.std(self.sy)
        self.sx /= self.std_sx
        self.sy /= self.std_sy

    def __getitem__(self, index):
        """Returns the sample indexed by the passed index."""
        target = np.hstack((self.sx[index], self.sy[index]))
        features = np.concatenate((self.psi[index], self.psi[index+1]), axis=0)
        return (features, target)

    def __len__(self):
        # -1 due to the fact that we use the psi at two time steps to make
        # predictions.
        return self.data_length - 1

    @property
    def n_output_targets(self):
        return self._n_output_targets

    def plot_true_vs_pred(self, true: np.ndarray, predicted: np.ndarray):
        true_sx, true_sy = np.split(true, 2, axis=1)
        pred_sx, pred_sy = np.split(predicted, 2, axis=1)
        fig = plt.figure()
        plt.subplot(321)
        plt.imshow(self.flat_to_2d(true_sx), vmin=-1, vmax=1, origin='lower',
                   cmap='jet')
        plt.title('true S_x')
        plt.subplot(322)
        plt.imshow(self.flat_to_2d(true_sy), vmin=-1, vmax=1, origin='lower',
                   cmap='jet')
        plt.title('true S_y')
        plt.subplot(323)
        plt.imshow(self.flat_to_2d(pred_sx), vmin=-1, vmax=1, origin='lower',
                   cmap='jet')
        plt.title('predicted S_x')
        plt.subplot(324)
        plt.imshow(self.flat_to_2d(pred_sy), vmin=-1, vmax=1, origin='lower',
                   cmap='jet')
        plt.title('predicted S_y')
        plt.subplot(325)
        plt.imshow(self.flat_to_2d(pred_sx - true_sx), vmin=-1, vmax=1,
                   origin='lower', cmap='jet')
        plt.title('difference')
        plt.subplot(326)
        plt.imshow(self.flat_to_2d(pred_sy - true_sy), vmin=-1, vmax=1,
                   origin='lower', cmap='jet')
        plt.title('difference')
        return fig

    def flat_to_2d(self, data):
        return data.reshape(self.width, self.height)


# Define a Neural Network
class SimpleNN(Module):
    def __init__(self, input_width: int, input_height: int, output_size: int):
        super().__init__()
        self.output_size = output_size
        self.conv1 = torch.nn.Conv2d(2, 128, 5, padding=2)
        self.conv2 = torch.nn.Conv2d(128, 64, 5, padding=2)
        self.conv3 = torch.nn.Conv2d(64, 32, 5, padding=2, groups=2)
        self.conv4 = torch.nn.Conv2d(32, 2, 5, padding=2, groups=2)
        self.state = NN_State.TRAIN

    def forward(self, input):
        out1 = F.selu(self.conv1(input))
        out2 = F.selu(self.conv2(out1))
        out3 = F.selu(self.conv3(out2))
        out4 = self.conv4(out3)
        out4 = out4.view(-1, self.output_size)
        return out4

    def set_for_test(self):
        self.state == NN_State.TEST

    def set_for_train(self):
        self.state == NN_State.TRAIN

#def weighted_mse(Y_pred: torch.Tensor, Y: torch.Tensor):
#    """Custom loss function based on a mse, but with weights corresponding to
#    the variance at each location"""
#    std_sx = torch.tensor(dataset.std_sx, device=device)
#    std_sy = torch.tensor(dataset.std_sy, device=device)
#    scaling = torch.mean(std_sx**2) + torch.mean(std_sy**2)
#    stds = torch.cat((std_sx, std_sy), dim=0)
#    return torch.mean((stds * (Y_pred - Y))**2) / scaling


# Define train and test data sets
# TODO how do I make sure the data is transfered to the GPU here if requested?
# Ie instead of having to do it manually later?
dataset = Dataset_psi_s('/data/ag7531/processed_data',
                        'psi_coarse.npy', 'sx_coarse.npy', 'sy_coarse.npy')
# Split the data into train and test sets. 3000 first points used for training
train_dataset = Subset(dataset, np.arange(3000))
test_dataset = Subset(dataset, np.arange(3000, 3049))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False)

# Put the NN on the GPU if possible
net = SimpleNN(128, 128, dataset.n_output_targets)
net.to(device)

# Define the MSE criterion and the Adam optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# Training
for e in range(nb_epochs):
    net.set_for_train()
    print('Epoch number %d' % e)
    running_loss = 0
    for i, data in enumerate(train_dataloader):
        # Zero the gradients
        net.zero_grad()
        # Get a batch and move it to the GPU (if possible)
        X = data[0].to(device, dtype=torch.float)
        Y = data[1].to(device, dtype=torch.float)
        loss = criterion(net(X), Y)
        running_loss += loss.item()
        if i % print_loss_every == print_loss_every - 1:
            # Every 'print_loss_every' batch we print the running loss
            running_loss /= print_loss_every
            print('Current loss value is %f' % (running_loss))
            running_loss = 0
        # Backpropagate 
        loss.backward()
        optimizer.step()
    # Log the training loss
    mlflow.log_metric('train mse', running_loss/50, e)
    
    # At the end of each epoch we compute the test loss and print it
    net.set_for_test()
    with torch.no_grad():
        nb_samples = 0
        running_loss = 0
        for i, data in enumerate(test_dataloader):
            X = data[0].to(device, dtype=torch.float)
            Y = data[1].to(device, dtype=torch.float)
            loss = criterion(net(X), Y)
            running_loss = running_loss + loss.item()
            nb_samples = i
    test_loss = running_loss / nb_samples
    print('Current test loss is %f' % test_loss)
    mlflow.log_metric('test mse', test_loss, e)
    
    # We also save a snapshot figure to the disk and log it
    id_data = np.random.randint(0, len(test_dataset))
    id_data = 25
    with torch.no_grad():
        data = test_dataset[id_data]
        X = torch.tensor(data[0][np.newaxis, ...]).to(device, dtype=torch.float)
        Y = data[1][np.newaxis, ...]
        pred = net(X).cpu().numpy()
        fig = dataset.plot_true_vs_pred(Y, pred)
        f_name = 'image{}.png'.format(e)
        file_path = os.path.join(data_location, figures_directory, f_name)
        plt.savefig(file_path)
        plt.close(fig)
        mlflow.log_artifact(file_path)
