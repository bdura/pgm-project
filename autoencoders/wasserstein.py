import torch
import torch.nn as nn
import torch.nn.functional as F


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def cost(x, y):
    return F.mse_loss(x, y)


def loss(x, x_tilde, z_tilde):

    z = torch.randn_like(z_tilde)
    pass


class WassersteinAutoEncoder(nn.Module):

    def __init__(self, ksi=10., hidden_dimension=2):
        super(WassersteinAutoEncoder, self).__init__()

        self.ksi = ksi
        self.hidden_dimension = hidden_dimension

        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 50)
        self.fc3 = nn.Linear(50, 12)
        self.fc4 = nn.Linear(12, hidden_dimension)

        self.fc5 = nn.Linear(hidden_dimension, 12)
        self.fc6 = nn.Linear(12, 50)
        self.fc7 = nn.Linear(50, 128)
        self.fc8 = nn.Linear(128, 28 * 28)

    def encode(self, x):

        x = x.view(-1, num_flat_features(x))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

    def decode(self, x):

        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        # We push the pixels towards 0 and 1
        x = torch.sigmoid(self.fc8(x))

        return x

    def forward(self, x):
        z = self.encode(x)
        x_tilde = self.decode(z)

        return x_tilde, z

    def loss(self, x, x_tilde, z):

        n = x.size()[0]

        recon_loss = F.binary_cross_entropy(x_tilde, x)

        z_fake = torch.randn(n, self.hidden_dimension)

        kernel_zf_zf = self.kernel(z_fake, z_fake)
        kernel_z_z = self.kernel(z, z)
        kernel_z_zf = self.kernel(z, z_fake)

        mmd_loss = ((1 - torch.eye(n)) * (kernel_zf_zf + kernel_z_z)).sum() / (n * (n-1)) - 2 * kernel_z_zf.mean()

        total_loss = recon_loss + self.ksi * mmd_loss

        return total_loss

    def kernel(self, x, y):
        """
        Returns a matrix K where :math:`K_{i, j} = k(x_i, y_j)

        Args:
            x: a PyTorch Tensor
            y: a PyTorch Tensor

        Returns:
            The kernel computed for every pair of x and y.
        """

        assert x.size() == y.size()

        # We use the advised constant, with sigma=1
        # c is the expected square distance between 2 vectors sampled from Pz
        c = self.hidden_dimension * 2

        x_ = x.unsqueeze(0)
        y_ = y.unsqueeze(1)

        return c / (c + (x_ - y_).pow(2).sum(2))


if __name__ == '__main__':
    model = WassersteinAutoEncoder()
    model.train()
