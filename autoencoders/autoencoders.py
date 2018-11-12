import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets


class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 50)
        self.fc3 = nn.Linear(50, 12)
        self.fc4 = nn.Linear(12, 2)

        self.fc5 = nn.Linear(2, 12)
        self.fc6 = nn.Linear(12, 50)
        self.fc7 = nn.Linear(50, 128)
        self.fc8 = nn.Linear(128, 28 * 28)

    def encode(self, x):
        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        return x

    def decode(self, x):
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        # We push the pixels towards 0 and 1
        x = F.sigmoid(self.fc8(x))

        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)

        torch.randn

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class VariationalAutoEncoder(nn.Module):

    def __init__(self):
        super(VariationalAutoEncoder, self).__init__()

        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 50)
        self.fc3 = nn.Linear(50, 12)
        self.fc41 = nn.Linear(12, 2)
        self.fc42 = nn.Linear(12, 2)

        self.fc5 = nn.Linear(2, 12)
        self.fc6 = nn.Linear(12, 50)
        self.fc7 = nn.Linear(50, 128)
        self.fc8 = nn.Linear(128, 28 * 28)

    def encode(self, x):
        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        mu, log_variance = self.fc41(x), self.fc42(x)

        return mu, log_variance

    def sample(self, mu, log_variance):

        sigma = torch.exp(.5 * log_variance)
        epsilon = torch.randn_like(sigma)

        return mu + sigma * epsilon

    def decode(self, x):
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        # We push the pixels towards 0 and 1
        x = F.sigmoid(self.fc8(x))

        return x

    def forward(self, x):
        mu, log_variance = self.encode(x)
        x = self.sample(mu, log_variance)
        x = self.decode(x)

        return x, mu, log_variance

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == '__main__':
    model = AutoEncoder()
    model.train()
