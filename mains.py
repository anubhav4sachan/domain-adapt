import torch

from net import mainNet
from datasets import get_source, get_target
from training import train, train_deep

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

source_train = get_source()
source_val = get_source(train = False)

target_train = get_target()
#target_val = get_target(train= False)

#train(source_train, target_train)
train_deep(source_train)