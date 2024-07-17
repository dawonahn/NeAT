
import torch
import torch.nn as nn

class CPD(nn.Module):
    ''' CP decomposition
    '''

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.sizes = cfg.sizes
        self.rank = cfg.rank
        self.mode = cfg.modes
        self.embeds = nn.ModuleList([nn.Embedding(self.sizes[i], self.rank)
                                     for i in range(len(self.sizes))])
        self._initialize()


    def _initialize(self):
        for i in range(len(self.embeds)):
            nn.init.uniform_(self.embeds[i].weight.data)
            

    def forward(self, idxs):
        
        embeds = [self.embeds[i](idxs[:, i]).unsqueeze(-1) 
                      for i in range(len(self.sizes))]
        x = torch.cat(embeds, dim=-1) # batch x rank x nmode
        x = torch.prod(x, dim=-1)    # batch x rank
        x = x.sum(-1)               # batch x 1

        return x


