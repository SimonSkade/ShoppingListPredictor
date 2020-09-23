from rscanner.model import RNN
from rscanner.util import randomChoice, lineToTensor
from rscanner.util import categoryFromOutput

import string
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# the dataset
with open('rscanner/data/c0.txt', 'r') as file:
    c0 = file.read()

with open('rscanner/data/c1.txt', 'r') as file:
    c1 = file.read()

with open('rscanner/data/c2.txt', 'r') as file:
    c2 = file.read()

training_data = [c0.upper().split(), c1.upper().split(), c2.upper().split()]

# a test dataset
test_data = 'l CITYBahnhofplatz  MnchenUID Nr DE K SPRUEHSAHNE EUR Stk x O MILCH E SCHOKOLADE P VANILLE D BUTTERMILCH R K VANILLESCHOTE  B MILCHSCHOKOSTR MILCHSCHOKOLADE BKLPAPIERTASCHE  A TRINKHALME  A'.upper().split()

n_hidden = 128

all_letters = string.ascii_letters + " .,;'" + "äÄüÜöÖ"
n_letters = len(all_letters)

rnn = RNN(n_letters, n_hidden, 3)

learning_rate = 0.005

criterion = nn.NLLLoss()

def randomTrainingExample():
    category = random.randint(0,2)
    line = randomChoice(training_data[category])

    category_tensor = torch.tensor([category], dtype=torch.long)

    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()
    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()

def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

def trainOnData():
    n_iters = 150000
    print_every = 5000
    plot_every = 1000

    current_loss = 0
    all_losses = []

    def timeSince(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    start = time.time()

    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output, loss = train(category_tensor, line_tensor)
        current_loss += loss

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess_i = categoryFromOutput(output)
            correct = '✓' if guess_i == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess_i, correct))

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0
    
#    print("\n Evaluating... \n")
#    for i in range(len(test_data)-1):
#        output = evaluate(lineToTensor(test_data[i]))
#        category = categoryFromOutput(output)
#        if category == 1:
#            print(test_data[i])

    torch.save(rnn.state_dict(), "rscanner/state_dicts/model.pt")