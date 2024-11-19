import torch.nn as nn
import torch
import numpy as np
import train as tr
##############################
import cv2
##################################

class TverskyLoss(nn.Module):

    def __init__(self):
        super(TverskyLoss, self).__init__()

    def forward(self, pred, gold):
        gold = torch.FloatTensor(gold[0]).cuda()
        pred = pred[0,0,:,:]

        inputs = torch.reshape(gold, (-1,))
        targets = torch.reshape(pred, (-1,))

        intersection = torch.sum(inputs * targets)

        fp = torch.sum((1 - targets) * inputs)
        fn = torch.sum(targets * (1 - inputs))
        tversky = torch.divide(
            intersection,
            intersection + fp * 0.6 + fn * 0.4 + 1e-8,
        )

        return 1 - tversky
