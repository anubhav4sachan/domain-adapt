import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
import random

from deepResNet import deepResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class mainNet(nn.Module):
    def __init__(self):
        super(mainNet, self).__init__()
        
        self.feature = deepResNet()
        
        self.domain = nn.Sequential(
                nn.Linear(1*80*80, 100),
                nn.ReLU(True),
                nn.Linear(100, 2),
                nn.Softmax()
                )
        
    def forward(self, x):
        x = x.expand(x.data.shape[0], 3, 80, 80)

        feat_r = self.feature(x)
        feat = feat_r.view(-1, 1*80*80)
        
        p = float(random.randint(1, 500)/10000)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
        rev_feat = ReverseLayerF.apply(feat, alpha)
        dom = self.domain(rev_feat)
        
        return feat_r, dom
#        return feat_r
    
#from torchsummary import summary
#model = mainNet().to(device)
#summary(model, (3, 80, 80))