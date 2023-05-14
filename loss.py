from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch.nn.functional as F
import torch as th
import numpy as np
import pickle as pkl
import json

class MMS_loss(th.nn.Module):
    def __init__(self, one_way=False):
        super(MMS_loss, self).__init__()
        self.one_way = one_way

    def forward(self, S, tokens=None, margin=0.001):
        deltas = margin * th.eye(S.size(0)).to(S.device)
        S = S - deltas

        target = th.LongTensor(list(range(S.size(0)))).to(S.device)
        if self.one_way:
            loss = 2 * F.nll_loss(F.log_softmax(S, dim=1), target)
        else:
            I2C_loss = F.nll_loss(F.log_softmax(S, dim=1), target)
            C2I_loss = F.nll_loss(F.log_softmax(S.t(), dim=1), target)
            loss = I2C_loss + C2I_loss
        return loss
