import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import models, transforms
import cv2
from skimage import io
from sklearn.model_selection import train_test_split
import matplotlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model = torch.load('./deep-model-0.pth')
model = torch.load('./model.pth')
model = model.eval()

Transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor(),
        ])


dataset = torchvision.datasets.ImageFolder(root='./data/tf', transform=Transform)

loader = torch.utils.data.DataLoader(dataset, batch_size=1)

def createParts(img):
    img_parts = []
    for i in range (0, 161, 80):
        for j in range (0, 241, 80):
            img_parts.append(img[i:i+80, j:j+80])
    return img_parts

def assemble(img_parts):
    img_cat = []        
    for i in range(0, 10, 4):
        arr = []
        for j in range (0, 1):
            f = np.hstack((img_parts[i], img_parts[i+1], img_parts[i+2], img_parts[i+3]))
            arr.append(f)
        img_cat.append(arr)
        
    r = np.vstack((img_cat[0][0], img_cat[1][0], img_cat[2][0]))
    return r
            
for data in loader:
    img_parts  = createParts(data[0].data.numpy().squeeze())
    img_p = np.asarray(img_parts)
    img_p = img_p[:, np.newaxis, :, :]
    out = model(torch.from_numpy(img_p).cuda())[0]
    fs = out.cpu().data.numpy().squeeze()
    arr = []
    for i in range (12):
        arr.append(fs[i,:, :])
    rs = assemble(arr)
    
#matplotlib.image.imsave('respore.png', rs)
matplotlib.image.imsave('domain.png', rs)