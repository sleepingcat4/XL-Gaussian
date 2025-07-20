import torch.nn as nn 
import torch 
import model 
from preprocessing import FuckBlood
from torchvision import datasets, transforms 
from torch.utils.data import Dataset, DataLoader 
import os
from tqdm import tqdm 
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.autoencoder(num_pixels=28*28).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

dataset = FuckBlood("IMAGE FOLDER PATH")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model.train()
for epoch in range(10):
    running_loss = 0
    dataloader_tqdm = tqdm(dataloader, desc=f"Epoch {epoch+1}/10", unit="batch")
    for batch_idx, batch in enumerate(dataloader_tqdm, 1):
        batch = batch.view(batch.size(0), -1).to(device)
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        avg_loss = running_loss / batch_idx
        dataloader_tqdm.set_postfix(loss=f"{avg_loss:.4f}")