# %load MNIST_GAN.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from models import Discriminator, Generator, TargetA 
import matplotlib.pyplot as plt
import sys
from time import sleep

#hypers
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4 
img_dim = 28 * 28 * 1
batch_size = 32
num_epochs = 1 #50

# initialize the models
disc = Discriminator(img_dim).to(device) 
gen = Generator(img_dim, img_dim).to(device)
target = TargetA().to(device)
PATH = "./trained_models/A/Trained_model_A"
target.load_state_dict(torch.load(PATH))
target.eval()

#set up the MNIST data
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
dataset = datasets.MNIST(root="./data", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# set up the optimizers and loss for the models
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
BCE_loss = nn.BCELoss() 
CE_loss = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for batch_idx, (real, labels) in enumerate(loader):
        #get a fixed input batch to display gen output
        if batch_idx == 0:
            if epoch == 0:
                fixed_input = real.view(-1,784).to(device)
        adv_ex = real.clone().reshape(-1,784).to(device)
        real = real.view(-1, 784).to(device) # [32, 784]
        batch_size = real.shape[0] # 32
        labels = labels.to(device) # size() [32]
        #make a copy of this batch to make examples
        #purturb each image in adv_ex
           
        for idx,item in enumerate(adv_ex):
            #item.size() = 784
            purturbation = gen(adv_ex[idx])
            adv_ex[idx] = adv_ex[idx] + purturbation #item.size() = 784   

            ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # where the second option of maximizing doesn't suffer from
        # saturating gradients
        output = disc(adv_ex).view(-1)
        lossG = torch.mean(torch.log(1. - output)) #get loss for gen's desired desc pred

        adv_ex = adv_ex.reshape(-1,1,28,28)
        f_pred = target(adv_ex)
        f_loss = CE_loss(f_pred, labels) #add loss for gens desired f pred
        loss_G_Final = f_loss+lossG # can change the weight of this loss term later

        opt_gen.zero_grad()
        loss_G_Final.backward()
        opt_gen.step()


    
        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))    
        disc_real = disc(real).view(-1)
        disc_fake = disc(adv_ex.detach()).view(-1)
        lossD = -torch.mean(torch.log(disc(real)) + torch.log(1. - disc(adv_ex)))
        # can decide later how much that loss term weighs
        
        opt_disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()


        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

                
torch.save(disc.state_dict(), "./disc_dict")
torch.save(gen.state_dict(), "./gen_dict")
