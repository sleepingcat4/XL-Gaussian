from torchvision import datasets, transforms 
from torch.utils.data import Dataset, DataLoader 
import os
from tqdm import tqdm 
from PIL import Image

class FuckBlood(Dataset):
  def __init__(self, folder_path):
    self.folder_path = folder_path 
    self.image_files = [f for f in tqdm(os.listdir(folder_path)) if f.lower().endswith((".png", ".jpg"))]
    self.transform = transforms.Compose([
      transforms.Resize((28, 28)),
      transforms.ToTensor()
    ])

  def __len__(self):
    return len(self.image_files)

  def __getitem__(self, idx):
    image_path = os.path.join(self.folder_path, self.image_files[idx])
    image = Image.open(image_path).convert("L") # opening in grayscale to reduce mem
    image = self.transform(image)
    return image