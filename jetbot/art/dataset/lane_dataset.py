# 使用公開數據集 (例：TuSimple)
import torch
from torch.utils.data import Dataset
from jetbot.art.auto_driving import transform

class LaneDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __getitem__(self, idx):
        img = transform(self.images[idx])
        mask = torch.LongTensor(self.masks[idx])
        return img, mask