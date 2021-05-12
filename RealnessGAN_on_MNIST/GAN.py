import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm

import os

from swd import swd
from scipy.stats import skewnorm


# Global Settings
SAVE_FOLDER = "resultsGAN"
NUM_EPOCHS = 100
BATCH_SIZE = 64
D_LR = 2e-4
G_LR = 2e-4
BETA_1 = 0.5
BETA_2 = 0.999

# Model Hyperparameters
LATENT_DIM = 20
HIDDEN_DIM = 256
IMAGE_DIM = 784



class Generator(nn.Module):
    def __init__(self, latent_size, hidden_size, out_size):
        super(Generator, self).__init__()

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


class Discriminator(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(Discriminator, self).__init__()

        self.L1 = nn.Linear(in_size, hidden_size)
        self.L2 = nn.Linear(hidden_size, hidden_size)
        self.L3 = nn.Linear(hidden_size, hidden_size)
        self.L4 = nn.Linear(hidden_size, 1)

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
        out = torch.sigmoid(out)
        return out



def train():
    # Device
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))

    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    local_datasets_dir = "G:/Users/Harry/Datasets"
    train_dataset = datasets.MNIST(local_datasets_dir, train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=8)

    # Load Networks
    d = Discriminator(IMAGE_DIM, HIDDEN_DIM).to(device)
    g = Generator(LATENT_DIM, HIDDEN_DIM, IMAGE_DIM).to(device)

    # Optimizer Settings
    d_optim = optim.Adam(d.parameters(), lr=D_LR, betas=[BETA_1, BETA_2])
    g_optim = optim.Adam(g.parameters(), lr=G_LR, betas=[BETA_1, BETA_2])

    criterion = nn.BCELoss()

    # Helper Functions
    def generate_latent(batch_size, latent_dim):
        return torch.empty(batch_size, latent_dim).uniform_(-1,1).to(device)

    fixed_z = generate_latent(64, LATENT_DIM)

    def scale(tensor, mini=-1, maxi=1):
        return tensor * (maxi - mini) + mini

    real_label = 1.
    fake_label = 0.

    def saveimg(image, savepath):
        image = image.transpose(1,2,0)
        plt.imsave(savepath, image)

    # Global Loss Logger
    losses = {"D": [], "G": []}
    swd_results = []
    
    # Train Proper
    for epoch in range(1, NUM_EPOCHS+1):
        print("========Epoch {}/{}========".format(epoch, NUM_EPOCHS))
        epoch_losses = {"D": [], "G": []}
        

        d.train()
        g.train()

        for real_images, real_labels in train_loader:
            original_images = real_images

            # Preprocess tensor
            batch_size = real_images.shape[0]
            real_images = real_images.view(batch_size, -1).to(device)
            real_images = scale(real_images, -1, 1)

            real_label_tensor = torch.full((batch_size,1), real_label, dtype=torch.float, device=device)
            fake_label_tensor = torch.full((batch_size,1), fake_label, dtype=torch.float, device=device)
           # print(real_label_tensor.shape)

            # Discriminator Real Loss
            d_optim.zero_grad()
            d_real_out = d(real_images)
            d_real_loss = criterion(d_real_out, real_label_tensor)
           
            # Discriminator Fake Loss
            z = generate_latent(batch_size, LATENT_DIM)
            fake_images = g(z)
            d_fake_out = d(fake_images)
            d_fake_loss = criterion(d_fake_out,fake_label_tensor)

            # Total Discriminator Loss, Backprop, and Gradient Descent
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optim.step()

            # Generator Forward Prop
            g_optim.zero_grad()
            z = generate_latent(batch_size, LATENT_DIM)
            g_images = g(z)
            d_g_out = d(g_images)

            # Generator Loss
            g_loss = criterion(d_g_out,real_label_tensor)

            # Total Generator Loss, Backprop and Gradient Descent
            g_loss.backward()
            g_optim.step()

            # Record Epoch Losses
            epoch_losses["D"].append(d_loss.item())
            epoch_losses["G"].append(g_loss.item())

        # Record Mean Epoch Losses
        losses["D"].append(np.mean(epoch_losses["D"]))
        losses["G"].append(np.mean(epoch_losses["G"]))
        print("D loss: {} G loss: {}".format(d_loss.item(), g_loss.item()))

        # Generate sample fake images after each epoch
        g.eval()
        with torch.no_grad():
            sample_tensor = g(fixed_z)
            sample_tensor = sample_tensor.view(-1, 1, 28, 28)

        original_images = original_images.expand(32, 3, 28, 28)
        fake_images = sample_tensor[:32].expand(32, 3, 28, 28)

        out = swd(original_images, fake_images, device=device)

        swd_results.append(out.item())
        print('SWD:', out)
      
    print('SWD',swd_results)
    print('d_loss', losses['D'])
    print('g_loss', losses['G'])



if __name__ == '__main__':  
    if os.path.isdir("./"+SAVE_FOLDER) == False :  
         os.mkdir(SAVE_FOLDER)  
    train()