import cv2
import torch
import numpy as np
import os


for files in os.walk('./PoreGroundTruthSampleimage'):
    pass
    
for name in files[2]:   
    img = cv2.imread('./PoreGroundTruthSampleimage/'+name, 0)
    cv2.imwrite(('./PoreGroundTruthSampleimage/' + name.split('.')[0] + '.jpg'), img)