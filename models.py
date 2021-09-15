import torch
from torch import nn
from torchvision import datasets, utils
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

transform = transforms.Compose(
    [transforms.ToTensor()])



def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def flatten(x):
    return torch.flatten()

class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = nn.LeakyReLU(0.01)(x)
        x = self.fc2(x)
        x = nn.Sigmoid()(x)
        return x


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.fc1 = nn.Linear(z_dim,  256)
        self.fc2 = nn.Linear(256, img_dim)
        
    def forward(self,x):
        x = self.fc1(x)
        x = nn.LeakyReLU(0.01)(x)
        x = self.fc2(x)
        x = nn.Tanh()(x)
        return x


class TargetA(nn.Module):  # models A and B are used in Tramèr et al. (2017).
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 5)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.drop1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64*20*20, 128) #(image_width - Filter_size + 2 * Padding) / Stride + 1
        self.drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.drop1(x)
        # print(x.shape)
        x = x.view(-1, 64 * 20 * 20)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.drop2(x)
        x = self.fc2(x)
        x = F.softmax(x)
        return x


class TargetB(nn.Module):  # models A and B are used in Tramèr et al. (2017).
    def __init__(self):
        super().__init__()
        self.drop1 = nn.Dropout(0.2)
        self.conv1 = nn.Conv2d(1, 64, 8)
        self.conv2 = nn.Conv2d(64, 128, 6)
        self.conv3 = nn.Conv2d(128, 128, 5)
        self.drop2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 12 * 12, 10) #(image_width - Filter_size + 2 * Padding) / Stride + 1

    def forward(self, x):
        x = nn.Dropout(0.2)(x)
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
       # print(x.shape)
        x = nn.Dropout(0.5)(x)
        x = x.view(-1, 128 * 12 * 12)
        x = self.fc1(x)
        x = F.softmax(x)
        return x



class TargetC(nn.Module):  # Model C is the target network architecture used in Carlini and Wagner (2017b)
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*4*4, 200)
        self.fc2 = nn.Linear(200, 10)



    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.conv4(x)
        x = nn.ReLU()(x)
        x = self.maxpool2(x)
        #print("c shape: " + str(x.shape))
        x = x.view(-1, 64*4*4)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = F.softmax(x)
        return x





