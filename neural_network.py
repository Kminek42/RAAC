import torch
import torch.nn as nn
import torch.fft as fft
import torchaudio.transforms as transforms
from matplotlib.pyplot import plot, show

class Encoder(nn.Module):
    def __init__(self, compress_ratio=8, hidden_size=256, transfer_bit_depth=4, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.activation = nn.Tanh()
        self.linear1 = nn.Linear(compress_ratio, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        self.bit_depth = transfer_bit_depth

    def reduce_bit_depth(self, input_tensor, bit_depth):
        max_value = 2**(bit_depth-1)  # -1 because one bit for sign
        input_tensor = input_tensor * max_value
        input_tensor = input_tensor + input_tensor.round().detach() - input_tensor.detach()
        
        normalized_tensor = input_tensor / max_value
        
        return normalized_tensor

    def encode(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.reduce_bit_depth(x, self.bit_depth)
        return x
    
class Decoder(nn.Module):
    def __init__(self, *args, memory_size=128, compress_ratio=8, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.activation = nn.Tanh()
        self.linear1 = nn.Linear(1, compress_ratio)
        self.rnn = nn.RNN(input_size=compress_ratio, hidden_size=memory_size, num_layers=1, batch_first=True)
        self.linear2 = nn.Linear(memory_size, compress_ratio)

    def decode(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x, _ = self.rnn(x)
        x = self.linear2(x)
        x = self.activation(x)
        return x

class RecurrentAutoencoder(nn.Module):
    def __init__(self, *args, compress_ratio, transfer_bit_depth, noise_level, memory_size, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = Encoder(compress_ratio=compress_ratio, transfer_bit_depth=transfer_bit_depth)
        self.decoder = Decoder(compress_ratio=compress_ratio, memory_size=memory_size)
        self.noise_level = noise_level
     
    def forward(self, x):
        compressed = self.encoder.encode(x)
        compressed = compressed + self.noise_level * torch.randn(compressed.shape)
        decompressed = self.decoder.decode(compressed)
        return decompressed
    
    def forward_clean(self, x):
        compressed = self.encoder.encode(x)
        decompressed = self.decoder.decode(compressed)
        return decompressed

class AmplitudeSpectrumLoss(nn.Module):
    def __init__(self, window_fn=torch.hann_window):
        super(AmplitudeSpectrumLoss, self).__init__()
        self.window_fn = window_fn

    def forward(self, input_signals, target_signals):
        # Apply window function to input and target signals
        window = self.window_fn(len(input_signals[0]))  # Assuming all signals in the batch have the same length
        input_signals = input_signals * window
        target_signals = target_signals * window

        # Compute the amplitude spectrum using the Fourier transform
        input_spectra = torch.log10(torch.abs(fft.fft(input_signals, dim=-1)) + 1e-7)
        target_spectra = torch.log10(torch.abs(fft.fft(target_signals, dim=-1)) + 1e-7)

        # input_spectra = torch.abs(fft.fft(input_signals, dim=-1))
        # target_spectra = torch.abs(fft.fft(target_signals, dim=-1))

        # plot(input_spectra[0].detach().numpy())
        # plot(target_spectra[0].detach().numpy())
        # show()

        # Calculate the mean squared error between the amplitude spectra
        loss = nn.MSELoss()(input_spectra, target_spectra)

        return loss

class MelSpectrogramLoss(nn.Module):
    def __init__(self, sample_rate=16000):
        super(MelSpectrogramLoss, self).__init__()
        self.mel_transform = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=32
        )
        self.mse_loss = nn.MSELoss()

    def forward(self, predicted_audio, target_audio):
        # Calculate mel spectrograms
        predicted_mel = self.mel_transform(predicted_audio)
        target_mel = self.mel_transform(target_audio)
        # Compute MSE loss between mel spectrograms
        loss = self.mse_loss(predicted_mel, target_mel)

        return loss
