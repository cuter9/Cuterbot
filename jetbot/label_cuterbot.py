import os
import json
import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
import platform
from zipfile import ZipFile
from utils.remote_files_management import download_files_from_remote

# 宣告路徑
# DATASET_DIR = "jetbot_dataset"
# RAW_JSON = os.path.join(DATASET_DIR, "labels_raw.json")
# FINAL_JSON = os.path.join(DATASET_DIR, "labels.json")

print(f"Platform Identifier: {platform.uname()}")
# The dir_depo parameter can be set as you required:
if platform.uname().system == 'Windows':
    # dir_depo = 'D:\\AI_Lecture_Demos\\Data_Repo\\Cuterbot_Repo'
    DATASET_DIR = 'D:\\AI_Lecture_Demos\\Data_Repo\\Cuterbot_Repo_Whole'
else:
    home_directory = Path.home()
    DATASET_DIR = os.path.join(home_directory, 'Data_Repo/Cuterbot_Repo')

os.makedirs(DATASET_DIR, exist_ok=True)

remote_nano_ip = "192.168.1.104"    # The ip address of Jetson Nanbo
remote_dir_datafile = "/home/cuterbot/Cuterbot/notebooks/road_following/dataset_xy/"  # the directory of datafiles on Jetson Nano for training
training_datafile = 'Jetbot_bbox_2024-12-25_04-45-04.zip'  # check the data file name is correct in the remote_dir_datafile

if not os.path.isfile(os.path.join(DATASET_DIR, training_datafile)):
    download_files_from_remote(remote_host=remote_nano_ip, remote_dir=remote_dir_datafile, remote_data_file=training_datafile, local_path=dir_depo)
else:
    print("The training datafile on the host exist and will be used for training")

with ZipFile(os.path.join(DATASET_DIR, training_datafile), 'r') as zObject:
    zObject.extractall(path=DATASET_DIR)



def run_auto_labeling():
    print("⏳ 正在載入預訓練 YOLOv8 模型進行自動標註...")
    # 使用官方預訓練的 YOLOv8x 或自訂在實驗室環境中微調過的物體偵測模型
    # 這裡以預訓練模型為例。實務上你也可以用專門偵測小車的 yolov8n.pt
    detector = YOLO("yolov8n.pt")

    if not os.path.exists(RAW_JSON):
        print(f"❌ 找不到原始資料庫索引檔 {RAW_JSON}，請先收集基礎巡航數據。")
        return

    with open(RAW_JSON, 'r') as f:
        raw_annotations = json.load(f)

    final_annotations = {}

    print(f"🚀 開始自動標註 {len(raw_annotations)} 張圖片...")
    for img_name, data in raw_annotations.items():
        img_path = os.path.join(DATASET_DIR, img_name)
        if not os.path.exists(img_path):
            continue

        # 讀取圖片並取得解析度
        img = cv2.imread(img_path)
        h, w, _ = img.shape

        # 執行 YOLO 偵測
        results = detector(img, verbose=False)[0]

        target_x, target_y, target_size = 0.0, 0.0, 0.0
        collision = 0

        # 解析偵測結果
        # 注意：若使用官方模型，JetBot 可能會被分類為 "sports ball", "toy" 或 "electronic device"
        # 建議在前方車輛貼上高對比標籤，或直接取畫面中信心度最高的目標框來簡化實驗
        best_box = None
        max_conf = 0.0

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf > max_conf:  # 挑選最自信的偵測物
                max_conf = conf
                best_box = box

        if best_box is not None and max_conf > 0.4:  # 門檻值設為 0.4
            # 取得邊界框像素座標 [xmin, ymin, xmax, ymax]
            xyxy = best_box.xyxy[0].tolist()

            # 計算中心點像素
            cx = (xyxy[0] + xyxy[2]) / 2.0
            cy = (xyxy[1] + xyxy[3]) / 2.0

            # 計算物體寬高
            box_w = xyxy[2] - xyxy[0]
            box_h = xyxy[3] - xyxy[1]

            # 🛠️ 關鍵：將座標歸一化到 [-1.0, 1.0] 的範圍，以匹配 EfficientNet 迴歸頭
            target_x = (cx / w) * 2.0 - 1.0
            target_y = (cy / h) * 2.0 - 1.0

            # 大小歸一化到 [0.0, 1.0] 之間（物體佔據畫面寬度的比例）
            target_size = box_w / w

            # 自動判定碰撞：若前車佔據畫面寬度超過 55%，視為過近
            if target_size > 0.55:
                collision = 1

        # 整合數據寫入全新標籤檔
        final_annotations[img_name] = {
            "steering": data["steering"],  # 保留原本手動點擊的路徑引導點
            "target": [target_x, target_y, target_size],  # YOLO 自動計算
            "collision": collision  # YOLO 自動判定
        }

    with open(FINAL_JSON, 'w') as f:
        json.dump(final_annotations, f, indent=4)
    print(f"✅ 自動標註完成！新標籤已儲存至 {FINAL_JSON}")


if __name__ == "__main__":
    run_auto_labeling()
