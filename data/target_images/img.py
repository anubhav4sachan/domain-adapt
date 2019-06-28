import cv2
import torch
import numpy as np
import os


for files in os.walk('./Train_500'):
    pass
    
for name in files[2]:   
    img = cv2.imread('./Train_500/'+name)    
    img_parts = []
    for i in range (0, 161, 80):
        for j in range (0, 241, 80):
            img_parts.append(img[i:i+80, j:j+80, :])
    f = 0
    for k in img_parts:
        f+=1
        cv2.imwrite('./train-aug/' + name.split('.')[0] + '_' +str(f) + '.jpg', k)

#img_cat = []        
#for i in range(0, 10, 4):
#    arr = []
#    for j in range (0, 1):
#        f = np.hstack((img_parts[i], img_parts[i+1], img_parts[i+2], img_parts[i+3]))
#        arr.append(f)
#    img_cat.append(arr)
#    
#r = np.vstack((img_cat[0][0], img_cat[1][0], img_cat[2][0]))
