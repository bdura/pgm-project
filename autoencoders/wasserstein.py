import torch
import torch.nn as nn
import torch.nn.functional as F


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
        """
        Encoder in the WAE architecture.

        Args:
            x: a Torch tensor, from the original space.

        Returns:
            the encoding of x in the latent space.
        """

        n = x.size(0)
        x = x.view(n, -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

    def decode(self, x):
        """
        Decoder in the WAE architecture

        Args:
            x: a Torch tensor, from the latent space.

        Returns:
            the reconstruction of x, from the latent space to the original space.
        """

        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        # We push the pixels towards 0 and 1
        x = torch.sigmoid(self.fc8(x))

        return x

    def forward(self, x):
        """
        Performs the forward pass:
        * encoding from the original space into the latent representation ;
        * reconstruction with loss in the original space.

        Args:
            x: a Torch tensor, from the original space.

        Returns:
            the reconstructed example.
        """
        z = self.encode(x)
        x_tilde = self.decode(z)

        return x_tilde, z

    def loss(self, x, x_tilde, z, device):
        """
        WAE loss with MMD divergence.

        Args:
            x: samples from the original space.
            x_tilde: reconstruction by the network.
            z: latent space representation.
            device: device instance on which to run the computations (cpu or gpu).

        Returns:
            the MMD-based loss.
        """

        n = x.size(0)

        recon_loss = F.binary_cross_entropy(x_tilde, x.view(n, -1))

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
        Returns a matrix K where :math:`K_{i, j} = k(x_i, y_j)`

        Here we use the inverse multiquadratics kernel.

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


def test(epoch, model, test_loader, device, writer):

    model.eval()
    test_loss = 0

    # We do not compute gradients during the testing phase, hence the no_grad() environment
    with torch.no_grad():

        for i, (data, _) in enumerate(test_loader):

            data = data.to(device)
            x_tilde, z = model(data)

            test_loss += model.loss(x_tilde=x_tilde, x=data, z=z, device=device).item()

            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], x_tilde.view(100, 1, 28, 28)[:n]])

                writer.add_image('reconstruction', comparison.cpu(), epoch)

    test_loss /= len(test_loader.dataset)
    print('>> Test set loss: {:.4f}'.format(test_loss))

    writer.add_scalar('data/test-loss', test_loss, epoch)


def train(epoch, model, optimizer, train_loader, device, writer):

    model.train()
    train_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):
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

    print('>> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

    writer.add_scalar('data/train-loss', train_loss / len(train_loader.dataset), epoch)


if __name__ == '__main__':
    # Mainly for debugging purposes

    wae = WassersteinAutoEncoder()
    wae.train()
