import torch
import torch.nn as nn
from neural_network import RecurrentAutoencoder, AmplitudeSpectrumLoss, MelSpectrogramLoss
from data_manager import SpeechDataset
from torch.utils.data import DataLoader
from matplotlib.pyplot import plot, show

dataset = SpeechDataset(filename='dataset/tadek_7_2min.wav', compression_ratio=3, buffer_len=4096, prediction_shift=1)

model = RecurrentAutoencoder(compress_ratio=3, transfer_bit_depth=4, noise_level=0.0, memory_size=256)

training_loader = DataLoader(
    dataset=dataset,
    batch_size=16,
    shuffle=True
)


criterion = AmplitudeSpectrumLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=8, gamma=0.565)

Y = []

best_loss = float('inf')
n = 5
for epoch in range(16):
    loss_sum = 0
    for i, data in enumerate(training_loader):
        inputs, labels = data
        optimizer.zero_grad()
        prediction = model.forward(inputs)
        batch_n = len(inputs)
        loss = criterion(prediction.view(batch_n, -1), labels.view(batch_n, -1))
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

    scheduler.step()
    Y.append(loss_sum / len(training_loader))
    print(epoch, loss_sum / len(training_loader))

torch.save(model, 'model.pt')

plot(Y)
show()
