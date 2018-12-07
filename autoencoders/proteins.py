import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset


class ProteinDataset(Dataset):

    def __init__(self, data):

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return torch.Tensor(self.data[item])


class WassersteinAutoEncoder(nn.Module):

    def __init__(self, ksi=10., hidden_dimension=2, loss_function=F.binary_cross_entropy):
        super(WassersteinAutoEncoder, self).__init__()

        self.ksi = ksi
        self.hidden_dimension = hidden_dimension

        self.loss_function = loss_function

        self.fc1 = nn.Linear(24 * 82, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, hidden_dimension)

        self.fc5 = nn.Linear(hidden_dimension, 32)
        self.fc6 = nn.Linear(32, 128)
        self.fc7 = nn.Linear(128, 512)
        self.fc8 = nn.Linear(512, 24 * 82)

    def encode(self, x):

        n = x.size(0)
        x = x.view(n, -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

    def decode(self, x):

        n = x.size(0)

        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)

        x = x.view(n, 24, 82)

        # We push the pixels towards 0 and 1
        x = F.softmax(x, dim=1)

        return x.view(n, 24 * 82)

    def forward(self, x):
        z = self.encode(x)
        x_tilde = self.decode(z)

        return x_tilde, z

    def log_softmax(self, x):

        x = self.encode(x)

        n = x.size(0)

        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)

        x = x.view(n, 24, 82)

        # We push the pixels towards 0 and 1
        x = F.log_softmax(x, dim=1)

        return x.view(n, 24 * 82)

    def loss(self, x, x_tilde, z, device):

        n = x.size(0)

        recon_loss = self.loss_function(x_tilde.view(n, -1), x.view(n, -1))

        z_fake = torch.randn(n, self.hidden_dimension).to(device)

        kernel_zf_zf = self.kernel(z_fake, z_fake)
        kernel_z_z = self.kernel(z, z)
        kernel_z_zf = self.kernel(z, z_fake)

        mmd_loss = ((1 - torch.eye(n).to(device)) * (kernel_zf_zf + kernel_z_z)).sum() \
                   / (n * (n-1)) - 2 * kernel_z_zf.mean()

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


class WAESinai(WassersteinAutoEncoder):

    def __init__(self, ksi=10., hidden_dimension=2, loss_function=F.binary_cross_entropy):

        super().__init__(ksi=ksi, hidden_dimension=hidden_dimension, loss_function=loss_function)

        self.dropout = nn.Dropout(.5)

    def encode(self, x):

        n = x.size(0)
        x = x.view(n, -1)

        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        x = F.elu(self.fc2(x))
        x = self.dropout(x)
        x = F.elu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)

        return x

    def decode(self, x):

        n = x.size(0)

        x = F.elu(self.fc5(x))
        x = self.dropout(x)
        x = F.elu(self.fc6(x))
        x = self.dropout(x)
        x = F.elu(self.fc7(x))
        x = self.dropout(x)
        x = self.fc8(x)

        x = x.view(n, 24, 82)

        # We push the pixels towards 0 and 1
        x = F.softmax(x, dim=1)

        return x.view(n, 24 * 82)

    def log_softmax(self, x):

        x = self.encode(x)

        n = x.size(0)

        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)

        x = x.view(n, 24, 82)

        # We push the pixels towards 0 and 1
        x = F.log_softmax(x, dim=1)

        return x.view(n, 24 * 82)


class WAESinai2(nn.Module):

    def __init__(self, ksi=10., hidden_dimension=2, loss_function=F.binary_cross_entropy):
        super(WAESinai2, self).__init__()

        self.ksi = ksi
        self.hidden_dimension = hidden_dimension

        self.loss_function = loss_function

        self.fc1 = nn.Linear(24 * 82, 250)
        self.fc2 = nn.Linear(250, 250)
        self.fc3 = nn.Linear(250, 250)
        self.fc4 = nn.Linear(250, hidden_dimension)

        self.bn1 = nn.BatchNorm1d(250)

        self.fc5 = nn.Linear(hidden_dimension, 250)
        self.fc6 = nn.Linear(250, 250)
        self.fc7 = nn.Linear(250, 250)
        self.fc8 = nn.Linear(250, 24 * 82)

        self.dropout = nn.Dropout(.7)

    def encode(self, x):

        n = x.size(0)
        x = x.view(n, -1)

        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        x = F.elu(self.fc2(x))
        x = self.bn1(x)
        x = F.elu(self.fc3(x))
        x = self.fc4(x)

        return x

    def decode(self, x):

        n = x.size(0)

        x = F.elu(self.fc5(x))
        x = F.elu(self.fc6(x))
        x = self.dropout(x)
        x = F.elu(self.fc7(x))
        x = self.fc8(x)

        x = x.view(n, 24, 82)

        # We push the pixels towards 0 and 1
        x = F.softmax(x, dim=1)

        return x.view(n, 24 * 82)

    def log_softmax(self, x):

        x = self.encode(x)

        n = x.size(0)

        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)

        x = x.view(n, 24, 82)

        # We push the pixels towards 0 and 1
        x = F.log_softmax(x, dim=1)

        return x.view(n, 24 * 82)

    def forward(self, x):
        z = self.encode(x)
        x_tilde = self.decode(z)

        return x_tilde, z

    def loss(self, x, x_tilde, z, device):

        n = x.size(0)

        recon_loss = self.loss_function(x_tilde.view(n, -1), x.view(n, -1))

        z_fake = torch.randn(n, self.hidden_dimension).to(device)

        kernel_zf_zf = self.kernel(z_fake, z_fake)
        kernel_z_z = self.kernel(z, z)
        kernel_z_zf = self.kernel(z, z_fake)

        mmd_loss = ((1 - torch.eye(n).to(device)) * (kernel_zf_zf + kernel_z_z)).sum() \
                   / (n * (n-1)) - 2 * kernel_z_zf.mean()

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


class VAESinai(nn.Module):

    def __init__(self, hidden_dimension=2):
        super(VAESinai, self).__init__()

        self.fc1 = nn.Linear(24 * 82, 250)
        self.fc2 = nn.Linear(250, 250)
        self.fc3 = nn.Linear(250, 250)

        self.fc41 = nn.Linear(250, hidden_dimension)
        self.fc42 = nn.Linear(250, hidden_dimension)

        self.bn1 = nn.BatchNorm1d(250)

        self.fc5 = nn.Linear(hidden_dimension, 250)
        self.fc6 = nn.Linear(250, 250)
        self.fc7 = nn.Linear(250, 250)
        self.fc8 = nn.Linear(250, 24 * 82)

        self.dropout = nn.Dropout(.7)

    def encode(self, x):
        n = x.size(0)
        x = x.view(n, -1)

        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        x = F.elu(self.fc2(x))
        x = self.bn1(x)
        x = F.elu(self.fc3(x))
        x = F.relu(self.fc3(x))

        mu, log_variance = self.fc41(x), self.fc42(x)

        return mu, log_variance

    def sample(self, mu, log_variance):

        sigma = torch.exp(.5 * log_variance)
        epsilon = torch.randn_like(sigma)

        return mu + sigma * epsilon

    def decode(self, x):

        n = x.size(0)

        x = F.elu(self.fc5(x))
        x = F.elu(self.fc6(x))
        x = self.dropout(x)
        x = F.elu(self.fc7(x))
        x = self.fc8(x)

        x = x.view(n, 24, 82)

        # We push the pixels towards 0 and 1
        x = F.softmax(x, dim=1)

        return x.view(n, 24 * 82)

    def log_softmax(self, x):

        mu, log_variance = self.encode(x)
        x = self.sample(mu, log_variance)

        n = x.size(0)

        x = F.elu(self.fc5(x))
        x = F.elu(self.fc6(x))
        x = self.dropout(x)
        x = F.elu(self.fc7(x))
        x = self.fc8(x)

        x = x.view(n, 24, 82)

        # We push the pixels towards 0 and 1
        x = F.log_softmax(x, dim=1)

        return x.view(n, 24 * 82)

    def forward(self, x):
        mu, log_variance = self.encode(x)
        x = self.sample(mu, log_variance)
        x = self.decode(x)

        return x, mu, log_variance


def test(epoch, model, test_loader, device, writer):
    model.eval()
    test_loss = 0

    # We do not compute gradients during the testing phase, hence the no_grad() environment
    with torch.no_grad():

        for i, data in enumerate(test_loader):

            data = data.to(device)
            x_tilde, z = model(data)

            test_loss += model.loss(x_tilde=x_tilde, x=data, z=z, device=device).item()

            # if i == 0:
            #     n = min(data.size(0), 8)
            #     comparison = torch.cat([data[:n].view(n, 1, 24, 82), x_tilde.view(100, 1, 24, 82)[:n]])
            #
            #     writer.add_image('reconstruction', comparison.cpu(), epoch)

    test_loss /= len(test_loader.dataset)
    # print('>> Test set loss: {:.4f}'.format(test_loss))

    writer.add_scalar('loss/test', test_loss, epoch)


def train(epoch, model, optimizer, train_loader, device, writer):
    model.train()
    train_loss = 0

    for batch_idx, data in enumerate(train_loader):
        # We move the mini-batch to the device (useful is using a GPU)
        data = data.to(device)

        # We initialize the gradients
        optimizer.zero_grad()

        # We compute the recontruction of x (x_tilde) and its encoding (z)
        x_tilde, z = model(data)

        # We compute the loss
        loss = model.loss(x_tilde=x_tilde, x=data, z=z, device=device)

        # Backpropagation
        loss.backward()

        # Updating the loss
        train_loss += loss.item()

        # Updating the parameters
        optimizer.step()

    # print('>> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

    writer.add_scalar('loss/train', train_loss / len(train_loader.dataset), epoch)


if __name__ == '__main__':

    wae = WassersteinAutoEncoder()
    wae.train()
