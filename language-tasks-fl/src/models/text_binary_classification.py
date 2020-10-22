import torch
import torch.nn as nn
import torch.nn.functional as F

class TextBinaryClassificationModel(nn.Module):
    
    def __init__(self, params):
        
        
        super().__init__()
        
        self.vocabSize = params['vocabSize']
        self.embeddingDim = params['embeddingDim']
        self.hiddenDim = params['hiddenDim']
        self.outputDim = params['outputDim']
        self.numLayers   = params['numLayers']
        self.bidirectional = params['bidirectional']
        self.padIdx = params['padIdx']
        
        self.embedding = nn.Embedding(self.vocabSize, self.embeddingDim, padding_idx = self.padIdx)
        
        self.lstm = nn.LSTM(self.embeddingDim, 
                           self.hiddenDim, 
                           num_layers=self.numLayers, 
                           bidirectional=self.bidirectional, 
                           dropout=params['dropout'], batch_first=True)
        
        if(self.bidirectional):
            self.fc = nn.Linear(2*self.hiddenDim , self.outputDim)
        else:
            self.fc = nn.Linear(self.hiddenDim , self.outputDim)
        
        self.dropout = nn.Dropout(0.5)#params['dropout'])
        
        self.criterion = nn.BCELoss()
        
        self.sig = nn.Sigmoid()
        
    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)
        #print(batch_size)
        
        # embeddings and lstm_out
        
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        
        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hiddenDim)
        
        # dropout and fully connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        #out  = F.log_softmax(out, dim=1)
        # sigmoid function
        sig_out = self.sig(out)
        
        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get last batch of labels
        
        # return last sigmoid output and hidden state
        return sig_out, hidden
    
    
    def initHidden(self, batchSize):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        train_on_gpu= True
        
        if(train_on_gpu):
            hidden = (weight.new(self.numLayers, batchSize, self.hiddenDim).zero_().cuda(),
                   weight.new(self.numLayers, batchSize, self.hiddenDim).zero_().cuda())
        else:
            hidden = (weight.new(self.numLayers, batchSize, self.hiddenDim).zero_(),
                   weight.new(self.numLayers, batchSize, self.hiddenDim).zero_())
        
        return hidden
   
        
        
