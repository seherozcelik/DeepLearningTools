import torch.nn as nn
import torch
import numpy as np
import train as tr
##############################
import cv2
##################################

class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, gold):
        gold = torch.FloatTensor(gold[0]).cuda()
        pred = pred[0,0,:,:]

        inputs = torch.reshape(gold, (-1,))
        targets = torch.reshape(pred, (-1,))

        intersection = torch.sum(inputs * targets)
        dice = torch.divide(
            2.0 * intersection,
            torch.sum(gold) + torch.sum(pred) + 1e-8,
        )

        return 1 - dice
