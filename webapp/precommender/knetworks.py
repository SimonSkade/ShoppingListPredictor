import numpy as np
from webapp.precommender.kmeans import kmeans
import torch
import torch.nn as nn
import math

from datetime import datetime, date, timedelta, time

from torch.utils.tensorboard import SummaryWriter

# The LSTM model for every centroid networks
class LSTM(nn.Module):
    def __init__(self, input_size=20, hidden_layer_size=1500, output_size=20):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear1 = nn.Linear(hidden_layer_size, math.ceil((5/4)*hidden_layer_size))
        self.linear2 = nn.Linear(math.ceil((5/4)*hidden_layer_size), output_size)
        
        self.dropout = nn.Dropout(p=0.2)
        
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        drop_out = self.dropout(lstm_out)
        predictions = self.linear1(drop_out.view(len(input_seq), -1))
        predictions = self.linear2(predictions)
        return predictions[-1]

# A wrapper class for the LSTM model implementing teacher enforce learning 
class Network:
    def __init__(self, vocabSize, hidden_layer_size=1500, lr=0.0001, tw=4, device=torch.device("cpu")):
        super().__init__()
        
        assert (tw != 0), "The training window has to be bigger than 0!"
        
        self.hidden_layer_size = hidden_layer_size
        
        self.vocabSize = vocabSize
        
        self.device = device
        
        self.model = LSTM(input_size=self.vocabSize, hidden_layer_size=self.hidden_layer_size, output_size=self.vocabSize).to(self.device)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        self.tw = tw
    
    # A class for creating the input tensors for the training with the defined training window
    def create_inout_sequences(self, input_data, tw=4):
        inout_seq = []
        for i in range(len(input_data)-tw):
            train_seq = input_data[i:i+tw]
            train_label = input_data[i+tw:i+tw+1]
            inout_seq.append((train_seq ,train_label))
        return inout_seq
    
    # The train main loop (teacher forcing)
    def train(self, tdata, epochs=700, verbose=False):
        self.trainingData = self.create_inout_sequences(torch.FloatTensor(tdata), self.tw)
        
        assert (len(tdata[0]) == self.vocabSize), "The number of features of the input tensor doesn't match the defined vocabSize!"
        
        if verbose:
            writer = SummaryWriter()
        
        for i in range(epochs):
            for seq, labels in self.trainingData:
                self.optimizer.zero_grad()
                self.model.hidden_cell = (torch.zeros(1, 1, self.model.hidden_layer_size).to(self.device),
                        torch.zeros(1, 1, self.model.hidden_layer_size).to(self.device))

                y_pred = self.model(seq.to(self.device))

                single_loss = self.loss_function(y_pred, labels.view(self.vocabSize).to(self.device))
                single_loss.backward()
                self.optimizer.step()
                
            if verbose:
                writer.add_scalar('Loss/train', single_loss.item(), i)
    
    # A class for making a prediction based on the recent n datapoints
    def predict(self, data, future=1):
        assert (len(data[0]) == self.vocabSize), "The number of features of the input tensor doesn't match the defined vocabSize!"
        inputList = data[-self.tw:,:]
        inputList = inputList.tolist()
        for i in range(future):
            seq = torch.FloatTensor(inputList[-self.tw:]).to(self.device)
            with torch.no_grad():
                self.model.hidden_cell = (torch.zeros(1, 1, self.model.hidden_layer_size).to(self.device),
                        torch.zeros(1, 1, self.model.hidden_layer_size).to(self.device))
                inputList.append(self.model(seq).cpu().numpy())
        return inputList[-future:]

class knetworks:
    def __init__(self, k, data, vocabSize, device=torch.device("cpu")):
        super(knetworks, self).__init__()
        
        self.km = kmeans(k)
        self.k = k
        self.centroids = []
        
        self.data = data
        
        self.D = []
        self.W = []
        
        self.vocabSize = vocabSize
        
        self.networks = []
        
        for _ in range(k):
            self.networks.append(Network(vocabSize, device=device))
        
    def sampleRandom(self, centroid):
        return np.random.choice(np.array(range(len(self.data))),p=self.W[centroid])
    
    def train(self, samples, epochs):
        print("We are going to train " + str(self.k) + " networks for " + str(epochs) + " epochs with " + str(samples) + " samples each.")
        for i in range(self.k):
            for s in range(samples):
                user = self.sampleRandom(i)
                print("[" + str(i) + "][" + str(s) + "->" + str(user) + "] Training..." , end="\r")
                self.networks[i].model.train()
                self.networks[i].train(self.data[user], epochs=epochs)
    
    def calcMean(self, data):
        n = len(data)
        mean = np.empty((len(data[0])))
        for i in range(len(data[0])):
            mean[i] = np.sum(data[:n,i])
        return mean/n
    
    def fit(self, max_iters=1, optimize=False, verbose=False):
        means = []
        for user in self.data:
            means.append(self.calcMean(user))
        means = np.array(means)
        
        self.km.fit(means, max_iters=max_iters, optimize=optimize, verbose=verbose)
        self.k = self.km.k
        self.centroids = self.km.centroids
        
        self.D = self.km.calcDistances(self.centroids, means)

        self.W = np.minimum((1/(self.D+0.001)**2), np.full(self.D.shape, 50))

        self.W = np.array([self.W[i]/sum(self.W[i]) for i in range(self.k)])
    
    def save(self, filepath):
        # save the model state_dicts
        for i, net in enumerate(self.networks):
            torch.save(net.model.state_dict(), filepath + "/models/CN_" + str(i) + ".pth")
        
        # save the centroids array
        np.savetxt(filepath + '/centroids.csv', self.centroids, delimiter=',')
        # save the distances array
        np.savetxt(filepath + '/distances.csv', self.D, delimiter=',')
        # save the weights array
        np.savetxt(filepath + '/weights.csv', self.W, delimiter=',')

    
    def load(self, filepath):
        # load the model state_dicts
        for i,net in enumerate(self.networks):
            net.model.load_state_dict(torch.load(filepath + "/models/CN_" + str(i) + ".pth"))
            
        # load the centroids array
        self.centroids = np.loadtxt(filepath + '/centroids.csv', delimiter=',')
        # load the distances array
        self.D = np.loadtxt(filepath + '/distances.csv', delimiter=',')
        # load the weights array
        self.W = np.loadtxt(filepath + '/weights.csv', delimiter=',')
        
        self.k = len(self.centroids)
    
    def predict(self, data, future=1):
        mean = self.calcMean(data)
        distances = self.km.calcDistances(self.centroids, mean)
        weights = np.minimum((1/distances**2), np.full(distances.shape, 50))
        weights = np.array([weights[i]/sum(weights) for i in range(self.k)])
        
        prediction = []
        for i, net in enumerate(self.networks):
            net.model.eval()
            prediction.append(weights[i] * net.predict(data, future))
            
        return np.sum(np.array(prediction), axis=0) 