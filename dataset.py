import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import json
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

class OpenForensics(Dataset):
    def __init__(self, dataframe, root_dir,preloaded_splits,transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        if(preloaded_splits==True):
            self.load_split()
            self.loaders()
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]
        img_path = os.path.join(self.root_dir, str(img_name)+".png")
        image = Image.open(img_path)
        label = torch.tensor(int(self.dataframe.iloc[idx, 2]))  
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def split(self,train_size_percent, val_size_percent, test_size_percent):
        train_size = int(train_size_percent * len(self))
        val_size = int(val_size_percent * len(self))
        test_size = len(self) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(self, [train_size, val_size, test_size])
    
    def load_split(self):
        train_indices = pd.read_csv(os.path.join("Split","train_indices.csv")).values.flatten()
        self.train_dataset = Subset(self, train_indices)
        val_indices = pd.read_csv(os.path.join("Split","val_indices.csv")).values.flatten()
        self.val_dataset = Subset(self, val_indices)
        test_indices = pd.read_csv(os.path.join("Split","test_indices.csv")).values.flatten()
        self.test_dataset = Subset(self, test_indices)
        self.test_indices = pd.read_csv(os.path.join("Split","test_indices.csv"))

    def loaders(self):
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)
    


class Transformer():
    def __init__(self, size):
        self.transform = transforms.Compose([
        transforms.Resize((size, size)),  # Ridimensiona l'immagine a 224x224
        transforms.ToTensor(),           # Converte l'immagine in un tensore
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomRotation(10),
        #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizza le immagini
        ])