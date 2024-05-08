import torch
import torch.nn as nn
from layers.RevIN import RevIN
from models.SNN_layers.spike_neuron import *
from models.SNN_layers.spike_dense import *
from models.SNN_layers.spike_rnn import *

class Model(nn.Module):
    """
     VanillaRNN is the most direct and traditional method for time series prediction using RNN-class methods.
     It completes multi-variable long time series prediction through multi-variable point-wise input and cyclic prediction.
     """
    def __init__(self, configs):
        super(Model, self).__init__()

        # get parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.rnn_type = configs.rnn_type

        # build model
        assert self.rnn_type in ['rnn', 'gru', 'lstm', 'dhrnn']
        if self.rnn_type == "rnn":
            self.rnn = nn.RNN(input_size=self.enc_in, hidden_size=self.d_model, num_layers=1, bias=True,
                              batch_first=True, bidirectional=False)
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(input_size=self.enc_in, hidden_size=self.d_model, num_layers=1, bias=True,
                              batch_first=True, bidirectional=False)
        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=self.enc_in, hidden_size=self.d_model, num_layers=1, bias=True,
                              batch_first=True, bidirectional=False)
        elif self.rnn_type == 'dhrnn':
            self.rnn = DH_RNN(input_size=self.enc_in, hidden_size=self.d_model, num_layers=1, bias=True,
                              batch_first=True, bidirectional=False)

        self.predict = nn.Sequential(
            nn.Linear(self.d_model, self.enc_in)
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        x = x_enc # b,s,c

        # encoding
        if self.rnn_type == "lstm":
            _, (hn, cn) = self.rnn(x)
        else:
            _, hn = self.rnn(x) # b,s,d  1,b,d

        # decoding
        y = []
        if self.rnn_type == "lstm":
            for i in range(self.pred_len):
                yy = self.predict(hn)  # 1,b,c
                yy = yy.permute(1, 0, 2)  # b,1,c
                y.append(yy)
                _, (hn, cn) = self.rnn(yy, (hn, cn))
        else:
            for i in range(self.pred_len):
                yy = self.predict(hn)    # 1,b,c
                yy = yy.permute(1,0,2) # b,1,c
                y.append(yy)
                _, hn = self.rnn(yy, hn)
        y = torch.stack(y, dim=1).squeeze(2) # bc,s,1

        return y



# TODO 写一个DHRNN的类，要和nn.RNN接口一致   20240506



# #DH-RNN model
# class DH_RNN(nn.Module):
#     def __init__(self, input_size:int,  hidden_size:int, num_layers:int, bias:bool, batch_first:bool=True, bidirectional:bool=False):
#         super(DH_RNN, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.bias = bias
#         self.batch_first = batch_first
#         self.bidirectional = bidirectional
        
#         #DH-SRNN layer
#         self.rnn_1 = spike_rnn_test_denri_nospike(self.input_size, self.hidden_size, tau_ninitializer = 'uniform',low_n = 0,high_n = 4,vth= 1,dt = 1,branch =4,device=device,bias=is_bias)
#         #readout layer
#         self.dense_2 = readout_integrator_test(self.hidden_size, self.hidden_size, dt = 1, device=device, bias=self.bias)
#         torch.nn.init.xavier_normal_(self.dense_2.dense.weight)
      
#         if self.bias:
#             torch.nn.init.constant_(self.dense_2.dense.bias, 0)

#     def forward(self, input):
#         input.to(device)
#         b,seq_length,input_dim = input.shape
#         #self.dense_1.set_neuron_state(b)
#         self.dense_2.set_neuron_state(b)
#         self.rnn_1.set_neuron_state(b)

        
#         output = torch.zeros(b, self.hidden_size).to(device)
 
#         for i in range(seq_length):

#             input_x = input[:,i,:].reshape(b,input_dim)
#             mem_layer1,spike_layer1 = self.rnn_1.forward(input_x)
#             #mem_layer2,spike_layer2 = self.rnn_2.forward(spike_layer1)
#             # mem_layer3,spike_layer3 = self.dense_2.forward(spike_layer2)
#             mem_layer2 = self.dense_2.forward(spike_layer1)
#             if i>0:
#                 output += mem_layer2

#         output = output/seq_length
#         return output




#! 检查是否接口一致
#DH-RNN model
class DH_RNN(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, num_layers:int, bias:bool, batch_first:bool=True, bidirectional:bool=False):
        super(DH_RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        # DH-SRNN layer
        self.rnn_1 = spike_rnn_test_denri_nospike(input_size, hidden_size, tau_ninitializer='uniform', 
                                                  low_n=0, high_n=4, vth=1, dt=1, branch=4, bias=bias)
        
        # readout layer
        self.dense_2 = readout_integrator_test(hidden_size, hidden_size, dt=1, bias=bias)
        torch.nn.init.xavier_normal_(self.dense_2.dense.weight)

        if bias:
            torch.nn.init.constant_(self.dense_2.dense.bias, 0)

    def forward(self, input, h_0=None):
        # input.to(device)
        input.to(spike_rnn_test_denri_nospike.dense.weight.device)
        if self.batch_first:
            input = input.transpose(0, 1)  # Convert batch_first to seq_first
        
        b, seq_length, input_dim = input.shape

        # Initialize hidden states if not provided
        if h_0 is None:
            h_0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), b, self.hidden_size, 
                              device=spike_rnn_test_denri_nospike.dense.weight.device)
        
        output = torch.zeros(seq_length, b, self.hidden_size, device=spike_rnn_test_denri_nospike.dense.weight.device)
        
        h_n = h_0
        for i in range(seq_length):
            input_x = input[i].reshape(b, input_dim)
            mem_layer1, spike_layer1 = self.rnn_1(input_x)
            mem_layer2 = self.dense_2(spike_layer1)
            
            output[i] = mem_layer2
            h_n = mem_layer2  # Update last hidden state

        if self.batch_first:
            output = output.transpose(0, 1)  # Convert seq_first back to batch_first
        
        return output, h_n







# class DH_RNN(nn.Module):
#     def __init__(self, input_size:int, hidden_size:int, num_layers:int, bias:bool, batch_first:bool=True, bidirectional:bool=False):
#         super(DH_RNN, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.bias = bias
#         self.batch_first = batch_first
#         self.bidirectional = bidirectional

#         # DH-SRNN layer
#         self.rnn_1 = spike_rnn_test_denri_nospike(input_size, hidden_size, tau_ninitializer='uniform', low_n=0, high_n=4, vth=1, dt=1, branch=4, bias=bias)
        
#         # readout layer
#         self.dense_2 = readout_integrator_test(hidden_size, hidden_size, dt=1, bias=bias)
#         torch.nn.init.xavier_normal_(self.dense_2.dense.weight)

#         if bias:
#             torch.nn.init.constant_(self.dense_2.dense.bias, 0)

#     def forward(self, input, h_0=None):
#         if self.batch_first:
#             input = input.transpose(0, 1)  # Convert batch_first to seq_first
        
#         b, seq_length, input_dim = input.shape

#         # Initialize hidden states if not provided
#         if h_0 is None:
#             h_0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), b, self.hidden_size, device=input.device)
        
#         output = torch.zeros(seq_length, b, self.hidden_size, device=input.device)
        
#         h_n = h_0
#         for i in range(seq_length):
#             input_x = input[i].reshape(b, input_dim)
#             mem_layer1, spike_layer1 = self.rnn_1(input_x)
#             mem_layer2 = self.dense_2(spike_layer1)
            
#             output[i] = mem_layer2
#             h_n = mem_layer2  # Update last hidden state

#         if self.batch_first:
#             output = output.transpose(0, 1)  # Convert seq_first back to batch_first
        
#         return output, h_n
