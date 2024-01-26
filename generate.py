import torch
import torch.nn as nn
from neural_network import RecurrentAutoencoder
from data_manager import SpeechDataset, numpy_to_wav
from torch.utils.data import DataLoader
from matplotlib.pyplot import plot, show
import numpy as np 

dataset = SpeechDataset(filename='dataset/lalka.wav', compression_ratio=3, buffer_len=4096, prediction_shift=1)

model = torch.load('model.pt')

output = model.forward(dataset.get_combined(0, 30)).view(-1).detach().numpy()
output = np.convolve(output, np.ones((3)) / 3)

plot(output)
show()

numpy_to_wav(output, 'rnn_output.wav', sample_rate=16000)

output = model.encoder.encode(dataset.get_combined(0, 30)).view(-1).detach().numpy()
import numpy as np
import zlib

print(set(np.round(output, 4)))
data = ((output+1)*8).astype(np.int8)
plot(data)
show()
compressed_array = zlib.compress(data.tobytes())
with open('compressed_array.npy', 'wb') as f:
    f.write(compressed_array)

with open('compressed_array.npy', 'rb') as f:
    compressed_data = f.read()

# Decompress the data and load it into a NumPy array
decompressed_data = zlib.decompress(compressed_data)
loaded_array = np.frombuffer(decompressed_data, dtype=np.int8)

plot(loaded_array)
show()