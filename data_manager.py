import torch
from torchaudio.functional import resample
from torch.utils.data import Dataset
import scipy.io.wavfile as wavfile
import numpy as np

def wav_to_tensor(file_path, sample_rate=44100):
    # Read the MP3 file and convert it to a WAV file
    rate, data = wavfile.read(file_path)
    
    # If the audio has multiple channels, take the first channel
    if len(data.shape) > 1:
        data = data[:, 0]

    # Normalize the audio data to the range [-1, 1]
    data = data / np.max(np.abs(data), axis=0)

    # Convert the numpy array to a PyTorch tensor
    tensor_data = torch.from_numpy(data).float()

    return tensor_data, rate

def numpy_to_wav(data, file_path, sample_rate=44100):
    scaled_data = np.int16(data / np.max(np.abs(data)) * 32767)
    wavfile.write(file_path, sample_rate, scaled_data)

class SpeechDataset(Dataset):
    working_sample_rate = 16000

    def __init__(self, train=True, *, filename, compression_ratio, buffer_len, prediction_shift) -> None:
        super().__init__()
        self.shift = prediction_shift
        audio_tensor, sample_rate = wav_to_tensor(filename)
        audio_tensor = resample(audio_tensor, sample_rate, self.working_sample_rate)
        reminder = len(audio_tensor) % (compression_ratio * buffer_len)
        audio_tensor = audio_tensor[:-reminder]
        audio_tensor = audio_tensor.view(-1, buffer_len, compression_ratio)
        
        audio_tensor = audio_tensor - audio_tensor.min()
        audio_tensor = audio_tensor / audio_tensor.max()
        audio_tensor = audio_tensor * 2
        audio_tensor = audio_tensor - 1

        self.data = audio_tensor
        self.buffer_len = buffer_len
        self.compression_ratio = compression_ratio

    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, index) -> any:
        return self.data[index][self.shift:], self.data[index][:-self.shift]

    def get_combined(self, t0, dt):
        return self.data[
            (t0 * self.working_sample_rate) // self.compression_ratio // self.buffer_len
            :((t0+dt) * self.working_sample_rate) // self.compression_ratio // self.buffer_len
        ]
