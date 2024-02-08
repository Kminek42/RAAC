import torch
import torch.nn as nn
from neural_network import RecurrentAutoencoder
from data_manager import SpeechDataset, numpy_to_wav
from torch.utils.data import DataLoader
from matplotlib.pyplot import plot, show
import numpy as np 

dataset = SpeechDataset(filename='dataset/lalka_train.wav', compression_ratio=3, buffer_len=4096, prediction_shift=1)

model = torch.load('model.pt')

audio_input = dataset.get_combined(0, 30)
output = model.forward(audio_input).view(-1).detach().numpy()
output = np.convolve(output, np.ones((3)) / 3)

plot(output)
show()

numpy_to_wav(output, 'rnn_output.wav', sample_rate=16000)

output = model.encoder.encode(dataset.get_combined(0, 30)).view(-1).detach().numpy()
print(sorted(set(output.round(2))))
import numpy as np
import zlib

data = (output*8).astype(np.int8)
plot(sorted(output))
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
