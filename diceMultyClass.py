import torch.nn as nn
import torch 
from torch import Tensor, einsum

class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, gold):
        gold = torch.FloatTensor(gold[0]).cuda()
        pred = pred[0,:,:,:]
        
        dice = 0
        for i in range(5): #five class (0:bg), online learning == one image at a time
            inputs = torch.reshape(gold[i,:,:], (-1,))
            targets = torch.reshape(pred[i,:,:], (-1,))

            intersection = torch.sum(inputs * targets)
            dice += torch.divide(
                2.0 * intersection,
                torch.sum(gold[i,:,:]) + torch.sum(pred[i,:,:]) + 1e-8,
            )

        return 1 - dice/5