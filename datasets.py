import torch
import os
import numpy as np
import cv2
import PIL
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# SOURCE DATALOADER            
def get_source(train=True):
    if train == True:        
        fil = []
        for files in os.walk('./data/deep_pore_90_labels'):
            fil.append(files[2])               
        file = []                
        for name in fil:
            for i in name:
                i = str(i)
                file.append([i, i.split('.')[0]])                        
        names = []
        lbl = []                 
        for f in file:
            names.append(f[0])
            lbl.append(f[1])
            
        class imgdata(torch.utils.data.Dataset):
            def __init__(self, data_dir, label_dir, img_name, istrain,
                         transform, labels=None):
                self.data_dir = data_dir
                self.label_dir = label_dir
                self.img_name = img_name
                self.istrain = istrain
                self.transform = transform
                self.labels = labels
                
            def __len__(self):
                return len(self.labels)
            
            def __getitem__(self, idx):
                patchimg = os.path.join(self.data_dir, self.img_name[idx])
                labelimg = os.path.join(self.label_dir, self.img_name[idx])
                
                pi = cv2.imread(patchimg)
                li = cv2.imread(labelimg)
                
                pi = PIL.Image.fromarray(pi)
                li = PIL.Image.fromarray(li)

                label = torch.from_numpy(np.array(int(self.img_name[idx].split('.')[0])))
                
                if self.istrain:
                    return self.transform(pi), label, self.transform(li)
         
        data_transform = transforms.Compose([
                transforms.Grayscale(),
#                transforms.Normalize((.5,),(.5,)),
                transforms.ToTensor()
                ])                     
        dataset = imgdata('./data/deep_pore_90_patch', './data/deep_pore_90_labels',
                          img_name=names, istrain=True, transform=data_transform,
                          labels=lbl)     
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)  
        
        return dataloader
    
    elif train == False:
        fil = []
        for files in os.walk('./data/source_test_images/PoreGroundTruthSampleimage'):
            fil.append(files[2])               
        file = []                
        for name in fil:
            for i in name:
                i = str(i)
                file.append([i, i.split('.')[0]])                        
        names = []
        lbl = []                 
        for f in file:
            names.append(f[0])
            lbl.append(f[1])
            
        class imgdata(torch.utils.data.Dataset):
            def __init__(self, data_dir, label_dir, img_name, istrain,
                         transform, labels=None):
                self.data_dir = data_dir
                self.label_dir = label_dir
                self.img_name = img_name
                self.istrain = istrain
                self.transform = transform
                self.labels = labels
                
            def __len__(self):
                return len(self.labels)
            
            def __getitem__(self, idx):
                patchimg = os.path.join(self.data_dir, self.img_name[idx])
                labelimg = os.path.join(self.label_dir, self.img_name[idx])
                
                pi = cv2.imread(patchimg)
                li = cv2.imread(labelimg)
                
                pi = PIL.Image.fromarray(pi)
                li = PIL.Image.fromarray(li)

                label = torch.from_numpy(np.array(int(self.img_name[idx].split('.')[0])))
                
                if self.istrain:
                    return self.transform(pi), label, self.transform(li)
                
        data_transform = transforms.Compose([
                transforms.Grayscale(),
#                transforms.Normalize((.5,),(.5,)),
                transforms.ToTensor()
                ])                       
        dataset = imgdata('./data/source_test_images/PoreGroundTruthSampleimage',
                          './data/source_test_images/pore_GT_images_label',
                          img_name=names, istrain=True, transform=data_transform,
                          labels=lbl)           
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)  
        
        return dataloader
    
# TARGET DATALOADER
def get_target(train=True):
    if train == True:
        fil = []
        for files in os.walk('./data/target_images/train-aug'):
            fil.append(files[2])
                
        file = []
               
        for name in fil:
            for i in name:
                i = str(i)
                file.append([i, i.split('.')[0]])
                    
        names = []
        lbl = [] 
            
        for f in file:
            names.append(f[0])
            lbl.append(f[1])    
            
        class imgdata(Dataset):
            def __init__(self, data_dir, img_name, istrain, transform, labels=None):
                self.data_dir = data_dir
                self.img_name = img_name
                self.istrain = istrain
                self.transform = transform
                self.labels = labels
                
            def __len__(self):
                return len(self.labels)
            
            def __getitem__(self, idx):
                patchimg = os.path.join(self.data_dir, self.img_name[idx])
                pi = cv2.imread(patchimg)
                               
                label = torch.from_numpy(np.array(int(self.img_name[idx].split('.')[0])))
                
                if self.istrain:
                    return self.transform(pi), label
         
        data_transform = transforms.Compose([
#                transforms.Normalize((.5,),(.5,)),
                transforms.ToTensor()
                ])       
                    
        dataset = imgdata('./data/target_images/train-aug',
                              img_name=names, istrain=True, transform=data_transform,
                              labels=lbl)
               
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
        return dataloader
    
    elif train == False:
        fil = []
        for files in os.walk('./data/target_images/test-aug'):
            fil.append(files[2])
                
        file = []
               
        for name in fil:
            for i in name:
                i = str(i)
                file.append([i, i.split('.')[0]])
                    
        names = []
        lbl = [] 
            
        for f in file:
            names.append(f[0])
            lbl.append(f[1])    
            
        class imgdata(Dataset):
            def __init__(self, data_dir, img_name, istrain, transform, labels=None):
                self.data_dir = data_dir
                self.img_name = img_name
                self.istrain = istrain
                self.transform = transform
                self.labels = labels
                
            def __len__(self):
                return len(self.labels)
            
            def __getitem__(self, idx):
                patchimg = os.path.join(self.data_dir, self.img_name[idx])
                pi = cv2.imread(patchimg)
                               
                label = torch.from_numpy(np.array(int(self.img_name[idx].split('.')[0])))
                
                if self.istrain:
                    return self.transform(pi), label
         
        data_transform = transforms.Compose([
#                transforms.Normalize((.5,),(.5,)),
                transforms.ToTensor()
                ])         
                    
        dataset = imgdata('./data/target_images/test-aug',
                              img_name=names, istrain=True, transform=data_transform,
                              labels=lbl)
               
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
        return dataloader