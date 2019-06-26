import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import models, transforms
import cv2
from skimage import io
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model = CNNModel()
#model.load_state_dict(torch.load('./DANN-15.pth'))
#model.eval()

model = torch.load('./mod.pth')
model = model.eval()

Transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        ])


dataset = torchvision.datasets.ImageFolder(root='./data/tt', transform=Transform)

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
#    img_parts  = createParts(data[0].data.numpy())
#    out = []
#    for inp in img_parts:
#        out.append(model(torch.from_numpy(inp)))
    img = data[0].cuda()
    out = model(img)
    
f = out[0].cpu().data.numpy().squeeze()
#cv2.imwrite('11.jpg', out.cpu().data.numpy().squeeze())