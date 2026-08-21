import os
import random
import shutil
from pathlib import Path


def dataset_splitter(src_dir, dest_dir, test_ratio=0.8):
    src = Path(src_dir)
    dest = Path(dest_dir)
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    # 1. 撈出所有圖片檔案（支援 jpg, jpeg, png 等常見格式）
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    files = [f for f in src.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]

    # 2. 隨機打亂檔案順序
    random.shuffle(files)

    # 3. 計算分割點
    split_idx = int(len(files) * test_ratio)
    test_files = files[:split_idx]
    train_files = files[split_idx:]

    # 4. 建立目標資料夾（因為沒有 category，直接建 train 和 test 即可）
    train_dir = os.path.join(dest, "train")
    test_dir = os.path.join(dest,"test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 5. 複製檔案（使用 copy 比較安全，不容易遇到權限被鎖定的問題）
    for f in train_files:
        shutil.copy(f, os.path.join(train_dir, f.name))
    for f in test_files:
        shutil.copy(f, os.path.join(test_dir, f.name))

    print(f"分割完成！總共 {len(files)} 張圖片。")
    print(f"-> 訓練集 (train): {len(train_files)} 張")
    print(f"-> 測試集 (test): {len(test_files)} 張")

# 使用範例：
# split_flat_dataset("./my_dataset", "./clean_output", train_ratio=0.8)


