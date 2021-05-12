# Torch Imports
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Other Imports
import os
import numpy as np
import matplotlib.pyplot as plt

from swd import swd
from scipy.stats import skewnorm
from torchvision import datasets, transforms

# Global Settings
OUT_FOLDER = "RGAN_output_images"
EPOCHS = 50
BATCH_SIZE = 64
LATENT_DIMENSIONS = 20
HIDDEN_DIMENSIONS = 256
IMAGE_DIMENSIONS = 784
NUM_OUTCOMES = 10

class RGAN_G(nn.Module):
    def __init__(self, latent_size, hidden_size, out_size):
        super(RGAN_G, self).__init__()

        self.L1 = nn.Linear(latent_size, hidden_size)
        self.L2 = nn.Linear(hidden_size, hidden_size)
        self.L3 = nn.Linear(hidden_size, hidden_size)
        self.L4 = nn.Linear(hidden_size, hidden_size)
        self.L5 = nn.Linear(hidden_size, out_size)
        self.BNorm1 = nn.BatchNorm1d(hidden_size)
        self.BNorm2 = nn.BatchNorm1d(hidden_size)
        self.BNorm3 = nn.BatchNorm1d(hidden_size)
        self.BNorm4 = nn.BatchNorm1d(hidden_size)
        self.output = nn.Tanh()

    def forward(self, x):
        # Layer 1
        out = self.L1(x)
        out = self.BNorm1(out)
        out = F.relu_(out)
        # Layer 2
        out = self.L2(out)
        out = self.BNorm2(out)
        out = F.relu_(out)
        # Layer 3
        out = self.L3(out)
        out = self.BNorm3(out)
        out = F.relu_(out)
        # Layer 4
        out = self.L4(out)
        out = self.BNorm4(out)
        out = F.relu_(out)
        # Layer 5
        out = self.L5(out)
        out = self.output(out)
        return out

class RGAN_D(nn.Module):
    def __init__(self, in_size, hidden_size, num_outcomes):
        super(RGAN_D, self).__init__()

        self.L1 = nn.Linear(in_size, hidden_size)
        self.L2 = nn.Linear(hidden_size, hidden_size)
        self.L3 = nn.Linear(hidden_size, hidden_size)
        self.L4 = nn.Linear(hidden_size, num_outcomes)

    def forward(self, x):
        # Layer 1
        out = self.L1(x)
        out = F.leaky_relu(out, 0.02)
        # Layer 2
        out = self.L2(out)
        out = F.leaky_relu(out, 0.02)
        # Layer 3
        out = self.L3(out)
        out = F.leaky_relu(out, 0.02)
        # Layer 4
        out = self.L4(out)
        out = F.softmax(out)
        return out

# Scale image values
def scale_image(tensor, min=-1, max=1):
    return tensor * (max - min) + min

# Make latent tensor
def latent(batch_size, latent_dim, device):
    return torch.empty(batch_size, latent_dim).uniform_(-1,1).to(device)

# Save output image
def saveimg(image, savepath):
    image = image.transpose(1,2,0)
    plt.imsave(savepath, image)

# Kullback–Leibler divergence
def KLD(P, Q):
    return torch.mean(torch.sum(P * (P/Q).log(), dim=1))

def train():
    swd_results = []
    # Set Device Mode
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))

    # Download and construct Dataset
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    data = datasets.MNIST('data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

    # Load GANS
    d = RGAN_D(IMAGE_DIMENSIONS, HIDDEN_DIMENSIONS, NUM_OUTCOMES).to(device)
    g = RGAN_G(LATENT_DIMENSIONS, HIDDEN_DIMENSIONS, IMAGE_DIMENSIONS).to(device)

    # Optimizers
    d_optim = optim.Adam(d.parameters(), lr=2e-4, betas=[0.5, 0.999])
    g_optim = optim.Adam(g.parameters(), lr=2e-4, betas=[0.5, 0.999])

    fixed_z = latent(64, LATENT_DIMENSIONS, device)
    
    # A0
    skew = skewnorm.rvs(-5, size=1000)
    count, bins = np.histogram(skew, NUM_OUTCOMES)
    anchor0 = count / sum(count)

    # A1
    skew = skewnorm.rvs(5, size=1000)
    count, bins = np.histogram(skew, NUM_OUTCOMES)
    anchor1 = count / sum(count)

    A0 = torch.from_numpy(np.array(anchor0)).to(device).float()
    A1 = torch.from_numpy(np.array(anchor1)).to(device).float()

    # Record losses
    losses = {"D": [], "G": []}

    # Train Models
    for epoch in range(1, EPOCHS+1):
        print("Epoch:"+str(epoch))
        epoch_losses = {"D": [], "G": []}

        d.train()
        g.train()

        for train_images, _ in train_loader:
            # Preprocess tensor
            original_images = train_images
            batch_size = train_images.shape[0]
            train_images = train_images.view(batch_size, -1).to(device)
            train_images = scale_image(train_images, -1, 1)

            # Discriminator Real Loss
            d_optim.zero_grad()
            d_real_out = d(train_images)
            d_real_loss = KLD(d_real_out, A1)
           
            # Discriminator Fake Loss
            z = latent(batch_size, LATENT_DIMENSIONS, device)
            fake_images = g(z)
            d_fake_out = d(fake_images)
            d_fake_loss = KLD(A0, d_fake_out)

            # Discriminator Loss, Backprop, and Gradient Descent
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optim.step()

            # Gen Forward Propagation
            g_optim.zero_grad()
            z = latent(batch_size, LATENT_DIMENSIONS, device)
            g_images = g(z)
            d_g_out = d(g_images)

            # Generator Loss - EQ 19 from paper
            d_out = d(train_images)
            g_loss = -KLD(A0, d_g_out) + KLD(d_out, d_g_out)    # -KL(A0 || D(G(z))) + KL(D(x) || D(G(z)))

            # Gen Loss, Backprop and Gradient Descent
            g_loss.backward()
            g_optim.step()

            # Epoch Losses
            epoch_losses["D"].append(d_loss.item())
            epoch_losses["G"].append(g_loss.item())

        # Mean Epoch Losses
        losses["D"].append(np.mean(epoch_losses["D"]))
        losses["G"].append(np.mean(epoch_losses["G"]))

        # Make fake images
        g.eval()
        with torch.no_grad():
            sample_tensor = g(fixed_z)
            sample_tensor = sample_tensor.view(-1, 1, 28, 28)
        
        original_images = original_images.expand(32, 3, 28, 28)
        fake_images = sample_tensor[:32].expand(32, 3, 28, 28)

        out = swd(original_images, fake_images, device='cuda')
        
        swd_results.append(out.item())
        print('SWD:', out)
      
    print(swd_results)
    print('d_loss', losses['D'])
    print('g_loss', losses['G'])

if __name__ == '__main__':  
    if os.path.isdir("./"+OUT_FOLDER) == False :  
         os.mkdir(OUT_FOLDER)  
    train()