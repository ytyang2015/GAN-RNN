import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model_type="rnn", n_layers=1):
        super(RNN, self).__init__()
        """
        Initialize the RNN model.

        You should create:
        - An Embedding object which will learn a mapping from tensors
        of dimension input_size to embedding of dimension hidden_size.
        - Your RNN network which takes the embedding as input (use models
        in torch.nn). This network should have input size hidden_size and
        output size hidden_size.
        - A linear layer of dimension hidden_size x output_size which
        will predict output scores.

        Inputs:
        - input_size: Dimension of individual element in input sequence to model
        - hidden_size: Hidden layer dimension of RNN model
        - output_size: Dimension of individual element in output sequence from model
        - model_type: RNN network type can be "rnn" (for basic rnn), "gru", or "lstm"
        - n_layers: number of layers in your RNN network
        """

        self.model_type = model_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        ####################################
        #          YOUR CODE HERE          #
        ####################################
        self.encoder = nn.Embedding(self.input_size, self.hidden_size)

        if model_type == "rnn":
            self.rnn = nn.RNN(hidden_size, hidden_size, n_layers)
        elif model_type == "lstm":
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)
        elif model_type == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)

        self.decoder = nn.Linear(self.hidden_size, self.output_size)
        ##########       END      ##########



    def forward(self, input, hidden):
        """
        Forward pass through RNN model. Use your Embedding object to create
        an embedded input to your RNN network. You should then use the
        linear layer to get an output of self.output_size.

        Inputs:
        - input: the input data tensor to your model of dimension (batch_size)
        - hidden: the hidden state tensor of dimension (n_layers x batch_size x hidden_size)

        Returns:
        - output: the output of your linear layer
        - hidden: the output of the RNN network before your linear layer (hidden state)
        """


        ####################################
        #          YOUR CODE HERE          #
        ####################################
        batch_size = input.size()[0]
        emb = self.encoder(input).view(1,batch_size,self.hidden_size)
        if self.model_type == "lstm":
            emb = emb.contiguous().view(emb.size(0), emb.size(1), -1)

        output, hidden = self.rnn(emb, hidden)
#         decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        decoded = self.decoder(output)
#         output = decoded.view(output.size(0), output.size(1), decoded.size(1))
        output = decoded.view(batch_size, -1)
        ##########       END      ##########


        return output, hidden

    def init_hidden(self, batch_size, device=None):
        """
        Initialize hidden states to all 0s during training.

        Hidden states should be initilized to dimension (n_layers x batch_size x hidden_size)

        Inputs:
        - batch_size: batch size

        Returns:
        - hidden: initialized hidden values for input to forward function
        """

        hidden = None

        ####################################
        #          YOUR CODE HERE          #
        ####################################

        weight = next(self.parameters())
        if self.model_type == 'LSTM':
            hidden = (weight.new_zeros(self.n_layers, batch_size, self.hidden_size),
                    weight.new_zeros(self.n_layers, batch_size, self.hidden_size))
        else:
            hidden = weight.new_zeros(self.n_layers, batch_size, self.hidden_size)


        ##########       END      ##########

        return hidden
