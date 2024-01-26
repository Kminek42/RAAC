import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(8, 256),
    nn.Linear(256, 1)
)

torch.save(model, 'model.pt')
