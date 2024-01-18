import torch
import torch.nn as nn

class RecurrentAutoencoder(nn.Module):
    def __init__(self, *args, compress_ratio, transfer_bit_depth, memory_size, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bit_depth = transfer_bit_depth
        self.activation = nn.Tanh()

        self.pre_encoder = nn.Linear(compress_ratio, compress_ratio),
        self.encoder_rnn = nn.RNN(input_size=compress_ratio, hidden_size=memory_size, num_layers=1)
        self.post_encoder = nn.Linear(memory_size, 1)

        self.pre_decoder = nn.Linear(1, compress_ratio)
        self.decoder_rnn = nn.RNN(input_size=compress_ratio, hidden_size=memory_size, num_layers=1)
        self.post_decoder = nn.Linear(memory_size, compress_ratio)

    def reduce_bit_depth(self, input_tensor, bit_depth):
        # Calculate the maximum value based on the desired bit depth
        max_value = 2**(bit_depth-1)  # -1 because one bit for sign
        input_tensor = input_tensor * max_value
        # Quantize the input tensor
        input_tensor = input_tensor + input_tensor.round().detach() - input_tensor.detach()
        
        # Normalize back to the original scale
        normalized_tensor = input_tensor / max_value
        
        return normalized_tensor
     
    def forward(self, x):
        x = self.pre_encoder(x)
        x = self.activation(x)
        x, _ = self.encoder_rnn(x)
        x = self.post_encoder(x)
        x = self.activation(x)
        x = self.reduce_bit_depth(x, self.bit_depth)
        
        x = self.pre_decoder(x)
        x = self.activation(x)
        x, _ = self.decoder_rnn(x)
        x = self.post_decoder(x)
        x = self.activation(x)
