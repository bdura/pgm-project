import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np

class WAE(nn.Module):
    def __init__(self, input_dim, dim_z, reg,prior_var):
        super(WAE,self).__init__()

        self.reg = reg
        self.dim_z = dim_z
        self.prior_var = prior_var

        h1 = int((input_dim - dim_z) / 2)
        h2 = int((h1 - dim_z)/2)
        h3 = int((h2 - dim_z)/2)

        h1 = 128 
        h2 = 50
        h3 = 12
        #Encoding net
        self.fc1 = nn.Linear(input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2,h3)
        self.fc4 = nn.Linear(h3,dim_z)

        #Decoding net
        self.fc5 = nn.Linear(dim_z,h3)
        self.fc6 = nn.Linear(h3, h2)
        self.fc7 = nn.Linear(h2, h1)
        self.fc8 = nn.Linear(h1,input_dim)

    
    def encode(self, x):
        x = x.view(-1,x[0].numel())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

    def decode(self,z):
        z = F.relu(self.fc5(z))
        z = F.relu(self.fc6(z))
        z = F.relu(self.fc7(z))
        z = torch.sigmoid(self.fc8(z))
        return z

    def forward(self, x):
        z = self.encode(x)
        x_r = self.decode(z)
        return x_r, z

    def kernel(self,x,y):
        c = 2*self.dim_z*self.prior_var
        
        assert x.size() == y.size()

        # We use the advised constant, with sigma=1
        # c is the expected square distance between 2 vectors sampled from Pz

        x_ = x.unsqueeze(0)
        y_ = y.unsqueeze(1)

        return c / (c + (x_ - y_).pow(2).sum(2))

    def cost(self,x,y):
        return F.mse_loss(x,y)

    def sample_prior(self, num):
        return (MultivariateNormal(loc =torch.zeros(self.dim_z), 
               covariance_matrix=self.prior_var * torch.eye(self.dim_z)).sample(sample_shape=torch.Size([num])))


    def loss_fn(self,x, x_rec, z, device):
        n = x.size()[0]
        z_p = self.sample_prior(n).to(device)
        rec_loss = F.binary_cross_entropy(x_rec,x.view(n,-1))

        kernel_zp_zp = self.kernel(z_p, z_p)
        kernel_z_z = self.kernel(z, z)
        kernel_z_zp = self.kernel(z, z_p)

        mmd_loss = ((1 - torch.eye(n).to(device)) * (kernel_zp_zp + kernel_z_z)).sum() / (n * (n-1)) - (2 * kernel_z_zp.mean())

        total_loss = rec_loss + self.reg * mmd_loss

        return total_loss

def train(epoch, wae, train_loader, device):
    
    wae.train()
    train_loss = 0
    
    for batch_idx, (data, _) in enumerate(train_loader):
        
        # We move the mini-batch to the device (useful is using a GPU)
        data = data.to(device)
        
        # We initialize the gradients
        optimizer.zero_grad()
        
        # We compute the recontruction of x (x_tilde) and its encoding (z)
        x_tilde, z = wae(data)
        
        # We compute the loss
        loss = wae.loss_fn(x_rec=x_tilde, x=data, z=z, device = device)
        
        # Backpropagation
        loss.backward()
        
        # Updating the loss
        train_loss += loss.item()
        
        # Updating the parameters
        optimizer.step()

    print('>> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

def test(epoch, wae, test_loader, device):
    
    wae.eval()
    test_loss = 0
    
    # We do not compute gradients during the testing phase, hence the no_grad() environment
    with torch.no_grad():
        
        for i, (data, _) in enumerate(test_loader):
            
            data = data.to(device)
            x_tilde, z = wae(data)
            
            test_loss += wae.loss_fn(x_rec=x_tilde, x=data, z=z, device = device).item()
                
                #writer.add_image('reconstruction', comparison.cpu(), epoch)

    test_loss /= len(test_loader.dataset)
    print('>> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    root = '../data'
    
    trans = transforms.Compose([
    transforms.ToTensor()
    ])

    # if not exist, download mnist dataset
    train_set = datasets.MNIST(root=root, train=True, transform=trans, download=True)
    test_set = datasets.MNIST(root=root, train=False, transform=trans, download=True)

    batch_size_train = 100
    batch_size_test = 100

    train_loader = torch.utils.data.DataLoader(
                    dataset=train_set,
                    batch_size=batch_size_train,
                    shuffle=True)
    test_loader = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size=batch_size_test,
                    shuffle=False)

    wae = WAE(reg=10, input_dim=28*28, dim_z=2, prior_var=1)


    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    wae.to(device)

    learning_rate = 1e-3
    optimizer = torch.optim.Adam(wae.parameters(), lr=learning_rate, weight_decay=1e-5)
    space = np.array(
    [
        [x, y] 
        for y in np.linspace(-1.5, 1.5, 16) 
        for x in np.linspace(-1.5, 1.5, 16)
    ], 
    dtype=np.float
    )

    space = torch.from_numpy(space).type(torch.FloatTensor)

    #Train model
    for epoch in range(100, 200):
    
        train(epoch,wae, train_loader,device)
        test(epoch, wae, test_loader, device)
    
    with torch.no_grad():
        
        # sample = torch.randn(64, 2).to(device)
        
        sample = space.to(device) # + .05 * torch.randn(16 * 16, 2).to(device)
        sample = wae.decode(sample)
        
        utils.save_image(
            sample.view(16 * 16, 1, 28, 28), 
            'results/vae-sample_' + str(epoch) + '.png', 
            nrow=16
        )
