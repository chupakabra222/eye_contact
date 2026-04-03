import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2
import pandas as pd
from PIL import Image
import numpy as np
import kagglehub
from tqdm import tqdm
path = kagglehub.dataset_download("pratikyuvrajchougule/eye-contact")
path2 = path + "/combined_dataset/cutted/"
class EyeContactDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.raw_cache = {}
        self.path_cache = {}



    def warmup(self):
        print("Caching images to RAM...")
        for i in tqdm(range(len(self.dataframe))):
            _ = self[i] 
        print(f"Done! Cached {len(self.raw_cache)} images.")
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        label = self.dataframe.iloc[idx]['Y']
        
        if idx not in self.path_cache:
            self.path_cache[idx] = path2 + self.dataframe.iloc[idx]['New_Path']
        img_path = self.path_cache[idx]
        
        if idx not in self.raw_cache:
            img = Image.open(img_path).convert('RGB')
            tr1 = v2.Compose([
                v2.Resize((256,256))
            ])
            img = tr1(img)
            self.raw_cache[idx] = img
        else:
            img = self.raw_cache[idx]
        
        if self.transform:
            img_transformed = self.transform(img)
        else:
            img_transformed = transforms.ToTensor()(img)
        
        return img_transformed, torch.tensor(label, dtype=torch.long)