import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, action_dim, hidden_dim = 256, observation_shape = None):
        super(Model, self).__init__()

        #CNN Layers
        self.convi = nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 4, stride = 2)


#CNN - Recognize Image
#FC Layers