import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util import randomChoice, lineToTensor, readFile, categoryFromOutput

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

    def getCategory(self, word):
        lineTensor = lineToTensor(word)
        hidden = rnn.initHidden()

        for i in range(lineTensor.size()[0]):
            output, hidden = rnn(lineTensor[i], hidden)

        category = categoryFromOutput(output)
        return category