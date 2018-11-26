import torch.nn as nn
import torch
import torch.nn.functional as functional
from torch.distributions.multivariate_normal import MultivariateNormal

class WAE(nn.Module):
    def __init__(self, input_dim, dim_z, reg,prior_var):
        super(WAE,self).__init__()

        self.reg = reg
        self.dim_z = dim_z
        self.prior_var = prior_var

        hidden_dim = input_dim - dim_z / 2

        #Encoding net
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim_z)

        #Decoding net
        self.fc3 = nn.Linear(dim_z, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    
    def encode(self, x):
        x = x.view(-1,x.numel())
        x= F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x

    def decode(self,z):
        z = F.relu(self.fc3(z))
        z = torch.sigmoid(self.fc4(z))
        return z

    def forward(self, x):
        return self.decode(self.encode(x))

    def kernel(self,x,y):
        C = self.dim_z*self.prior_var
        
        assert x.size() == y.size()

        # We use the advised constant, with sigma=1
        # c is the expected square distance between 2 vectors sampled from Pz
        c = self.hidden_dimension * 2

        x_ = x.unsqueeze(0)
        y_ = y.unsqueeze(1)

        return c / (c + (x_ - y_).pow(2).sum(2))

    def cost(self,x,y):
        return F.mse_loss(x,y)

    def sample_prior(self, num):
        return (MultivariateNormal(loc =torch.zeros(self.dim_z), 
               covariance_matrix=self.prior_var * torch.eye(self.dim_z)).sample(sample_shape=torch.Size(num,self.dim_z)))


    def loss_fn(self,x, x_rec, z):
        n = x.size()[0]
        z_p = self.sample_prior(n)
        rec_cost = (self.cost(x,x_rec)).mean()

        kernel_zf_p = self.kernel(z_p, z_p)
        kernel_z_z = self.kernel(z, z)
        kernel_z_zp = self.kernel(z, z_p)

        mmd_loss = ((1 - torch.eye(n)) * (kernel_zp_zp + kernel_z_z)).sum() / (n * (n-1)) - (2/n) * kernel_z_zf.mean()

        total_loss = recon_loss + self.reg * mmd_loss

        return total_loss