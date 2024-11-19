import torch.nn as nn
import torch 
from torch import Tensor, einsum

class TverskyLoss(nn.Module):

    def __init__(self):
        super(TverskyLoss, self).__init__()

    def forward(self, pred, gold):
        gold = torch.FloatTensor(gold[0]).cuda()
        pred = pred[0,:,:,:]
        
        tversky = 0
        for i in range(5): #five channel, online learning == one image at a time
            inputs = torch.reshape(gold[i,:,:], (-1,))
            targets = torch.reshape(pred[i,:,:], (-1,))

            intersection = torch.sum(inputs * targets)

            fp = torch.sum((1 - targets) * inputs)
            fn = torch.sum(targets * (1 - inputs))
            tversky += torch.divide(
                intersection,
                intersection + fp * 0.6 + fn * 0.4 + 1e-8,
            )

        return 1 - tversky/5    