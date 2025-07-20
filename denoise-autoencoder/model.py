import torch.nn as nn 
import torch 

def autoencoder(num_pixels:int):
  model = nn.Sequential(
    nn.Linear(num_pixels, 500),
    nn.ReLU(),
    nn.Linear(500, 300),
    nn.ReLU(),
    nn.Linear(300, 100),
    nn.ReLU(),
    nn.Linear(100, 300),
    nn.ReLU(),
    nn.Linear(300, 500),
    nn.ReLU(),
    nn.Linear(500, 784),
    nn.Sigmoid()
  )

  return model