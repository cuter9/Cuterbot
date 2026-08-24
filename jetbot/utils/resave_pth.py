import torch
from pathlib import Path
import torchvision

dir_pth_files = "D:\\AI_Lecture_Demos\\Data_Repo\\Cuterbot_Repo\\GPU\\model_repo\\road_following"
src = Path(dir_pth_files)

files = [f.name for f in src.glob("*preprocess*")]
# torch.serialization.safe_globals([torchvision.transforms._presets.ImageClassification])
torch.serialization.add_safe_globals([torchvision.transforms._presets.ImageClassification])
for f in files:
    prep_file = src / f
    pth_data = torch.load(prep_file)
    torch.save(pth_data.state_dict(), dir_pth_files)