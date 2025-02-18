# 使用 COCO 或 KITTI 格式數據
from jetbot.art.auto_driving import transform
class VehicleDataset(Dataset):
    def __init__(self, annotations):
        self.annots = annotations

    def __getitem__(self, idx):
        img, targets = load_annotation(self.annots[idx])
        return transform(img), targets