from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

class AgePredictionCNN(nn.Module):
    """
    Age prediction CNN model
    """

    def __init__(self):
        super().__init__()

        self.loss = nn.MSELoss()

        self.conv1 = nn.Conv3d(1, 4, 3, padding=1)
        self.conv2 = nn.Conv3d(4, 4, 3, padding=1)
        self.batchnorm1 = nn.BatchNorm3d(4)
        self.maxpool1 = nn.MaxPool3d(2)

        self.conv3 = nn.Conv3d(4, 8, 3, padding=1)
        self.conv4 = nn.Conv3d(8, 8, 3, padding=1)
        self.batchnorm2 = nn.BatchNorm3d(8)
        self.maxpool2 = nn.MaxPool3d(2)

        self.conv5 = nn.Conv3d(8, 16, 3, padding=1)
        self.conv6 = nn.Conv3d(16, 16, 3, padding=1)
        self.batchnorm3 = nn.BatchNorm3d(16)
        self.maxpool3 = nn.MaxPool3d(2)

        self.conv7 = nn.Conv3d(16, 32, 3, padding=1)
        self.conv8 = nn.Conv3d(32, 32, 3, padding=1)
        self.batchnorm4 = nn.BatchNorm3d(32)
        self.maxpool4 = nn.MaxPool3d(2)

        self.fc1 = nn.Linear(32 * 6 * 6 * 6, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 1)

       
    def forward(self, x: Tensor):
        """
        Forward pass of the CNN.
        :param x: Input tensor of shape (N, 1, H, W, D)
        """
        x = F.relu(self.conv1(x))  # (N, 4, 96, 96, 96)
        x = F.relu(self.conv2(x))  # (N, 4, 96, 96, 96)
        x = self.batchnorm1(x)     # (N, 4, 96, 96, 96)
        x = self.maxpool1(x)       # (N, 4, 48, 48, 48)

        x = F.relu(self.conv3(x))  # (N, 8, 48, 48, 48)
        x = F.relu(self.conv4(x))  # (N, 8, 48, 48, 48)
        x = self.batchnorm2(x)     # (N, 8, 48, 48, 48)
        x = self.maxpool2(x)       # (N, 8, 24, 24, 24)

        x = F.relu(self.conv5(x))  # (N, 16, 24, 24, 24)
        x = F.relu(self.conv6(x))  # (N, 16, 24, 24, 24)
        x = self.batchnorm3(x)     # (N, 16, 24, 24, 24)
        x = self.maxpool3(x)       # (N, 16, 12, 12, 12)

        x = F.relu(self.conv7(x))  # (N, 32, 12, 12, 12)
        x = F.relu(self.conv8(x))  # (N, 32, 12, 12, 12)
        x = self.batchnorm4(x)     # (N, 32, 12, 12, 12)
        x = self.maxpool4(x)       # (N, 32, 6, 6, 6)

        x = x.view(-1, 32 * 6 * 6 * 6)  # (N, 32 * 6 * 6 * 6)
        x = F.relu(self.fc1(x))         # (N, 2048)
        x = F.relu(self.fc2(x))         # (N, 1024)
        x = self.fc3(x)                 # (N, 1)

        return x

    def train_step(self, imgs: Tensor, labels: Tensor, return_prediction: Optional[bool] = False):
        """
        Perform a training step. Predict the age for a batch of images and
        return the loss.

        :param imgs: Batch of input images (N, 1, H, W, D)
        :param labels: Batch of target labels (N)
        :return loss: The current loss, a single scalar.
        :return pred
        """
        pred = torch.squeeze(self.forward(imgs.float()))  # (N)
        #print(pred)
        # ----------------------- ADD YOUR CODE HERE --------------------------
        
        loss = self.loss(pred, labels.float())
        #pred = pred.float()
        #print(loss)
        # ------------------------------- END ---------------------------------

        if return_prediction:
            return loss, pred
        else:
            return loss
