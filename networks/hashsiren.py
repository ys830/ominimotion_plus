import torch
from torch import nn
import numpy as np
############ SIREN Network ############
class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()

    def init_weights(self):
        b = 1 / \
            self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)
    
class HashSiren(nn.Module):
    def __init__(self,
                 hash_mod=True,
                 hash_table_length=256*256, 
                 in_features=2, 
                 hidden_features=256, 
                 hidden_layers=4, 
                 out_features=1,
                 outermost_linear=True, 
                 first_omega_0=30, 
                 hidden_omega_0=30.0):

        super().__init__()
        self.hash_mod = hash_mod
        self.out_features = out_features

        self.table = nn.parameter.Parameter(1e-4 * (torch.rand((hash_table_length,in_features))*2 -1),requires_grad = True)
        
        self.net = []
        self.net.append(SirenLayer(in_features, hidden_features, 
                                  is_first=True, w0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SirenLayer(hidden_features, hidden_features,
                                      is_first=False, w0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SirenLayer(hidden_features, out_features,
                                      is_first=False, w0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        sh = coords.shape
        coords = torch.reshape(coords, [-1,2])

        if self.hash_mod:
            output = self.net(self.table[:,:])
        else:
            output = self.net(coords)

        # output = torch.clamp(output, min = -1.0,max = 1.0)
        # m = nn.Sigmoid()
        # output = m(output)
        # output = torch.reshape(output, list(sh[:-1]))

        output_size = [[[[0 for _ in range(self.out_features)] for _ in range(sh[2])] for _ in range(sh[1])] for _ in range(sh[0])]
        output = torch.reshape(output, list(np.array(output_size).shape))

        return output