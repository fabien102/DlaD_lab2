import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import torch.utils.data as utils
import time
import pdb
from torch.utils.data.sampler import SubsetRandomSampler




class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        
        ################################################################################
        # TODO:                                                                        #
        # Define 2 or more different layers of the neural network                      #
        ################################################################################
      
        self.fc1 = torch.nn.Linear(n_feature, n_hidden)
        self.fc2 = torch.nn.Linear(n_hidden, n_output)
    
        #pass
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################


    def forward(self, x):
        #print(x.size())
        #x = x.view(x.size(0),-1)
       
        ################################################################################
        # TODO:                                                                        #
        # Set up the forward pass that the input data will go through.                 #
        # A good activation function betweent the layers is a ReLu function.           #
        ################################################################################
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        
        #pass
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        return x
    
    def predict(self, x):
        ''' This function for predicts classes by calculating the softmax '''
        logits = self.forward(x)
        return F.softmax(logits)


