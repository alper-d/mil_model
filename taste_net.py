from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torch


class taste_network(nn.Module):
  def __init__(self, bag_size=526, num_of_classes=3):
    self.RNN1_OUT_SIZE = 32
    self.FC1_OUT_SIZE = 16*7*7
    self.RNN2_OUT_SIZE = 8*7*7
    self.FC2_OUT_SIZE = 8
    self.OUTPUT_SIZE = num_of_classes
    super(taste_network, self).__init__()
    # self.LSTM_1 = nn.LSTM((bag_size,512,7,7), self.RNN_OUT_SIZE, batch_first=True)
    #self.ATTENTION_LAYER = nn.Linear(in_features=(512*7*7), out_features=(64*7*7))
    self.RNN1 = nn.LSTM(input_size=1, hidden_size=self.RNN1_OUT_SIZE,num_layers=1, batch_first=True)
    # self.FC_1 = nn.Linear(self.RNN1_OUT_SIZE, self.FC1_OUT_SIZE)
    # self.RNN2 = nn.RNN(input_size=self.FC1_OUT_SIZE, hidden_size=self.RNN2_OUT_SIZE,batch_first=True)
    self.FC_2 = nn.Linear(self.RNN1_OUT_SIZE, self.FC2_OUT_SIZE)
    self.OUT_LAYER = nn.Linear(self.FC2_OUT_SIZE, self.OUTPUT_SIZE)

  def forward(self, inp):
    # print(type(inp))
    # print(inp.__len__())
    # print(inp[0])
    # print(inp[1])
    # print(inp.data.is_cuda)
    inp = torch.nn.utils.rnn.pack_padded_sequence(inp[0],inp[1].cpu(),batch_first=True)
    #shp = inp.shape
    #print(shp)
    # inp = inp.view(shp[0],shp[1],-1)
    inp = self.RNN1(inp)
    # inp = self.FC_1(inp[0])
    # inp = self.RNN2(inp)
    inp = self.FC_2(inp[1][0][-1,...])
    inp = self.OUT_LAYER(inp)
    return inp
    