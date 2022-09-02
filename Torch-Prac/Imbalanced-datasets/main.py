import os
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Methods for dealing with imbalanced datasets:
# 1 Oversampling --> preferred
# 2 Class weighting


# For Class weighting
# for two classes, the second class' loss would be multiplied by 50, so essentially
# weight has been distributed according to the number of samples
# In this case number of samples in class[0] = 50 * num samples in class[1]
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1, 50]))
