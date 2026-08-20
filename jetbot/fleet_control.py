import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.models.detection as detection
import torchvision.models as models
import torchvision.transforms as tf
import numpy as np

# 強制設定 Matplotlib 使用 headless 模式，防止在遠端工作站或無顯示器環境下報錯
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 項目基礎設定
DATASET_DIR = "jetbot_dataset"
AUTO_JSON = os.path.join(DATASET_DIR, "labels_auto.json")
# 🎛️ 可選模型字串: 'mobilenetv4', 'ssdlite_v3', 'efficientdet', 'mobilevit', 'yolo26', 'yolov11', 'rf_detr'
MODEL_CHOICE = "yolo26"

# 嘗試初始化主機端顯示卡功耗監測器 (NVIDIA NVML)
try:
    import pynvml

    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    HAS_NVML = True
except Exception:
    HAS_NVML = False
    print("⚠️ 未檢測到 NVML 庫，主機訓練功耗將以模擬資料顯示。")


# ──────────────────────────────────────────────────────────
# 1. 訓練期高精度指標與收斂繪圖器 (Convergence & Metric Plotter)
# ──────────────────────────────────────────────────────────
class TrainingProgressTracker:
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.epoch_indices = []
        self.loss_history = []
        self.fps_history = []
        self.power_history = []

        # 初始化繪圖畫布 (左圖放綜合收斂曲線，右圖放硬體功耗與處理速度)
        self.fig, (self.ax_loss, self.ax_hardware) = plt.subplots(1, 2, figsize=(11, 4.5))
        self.report_path = "training_report.png"

    def append_epoch_metrics(self, epoch, loss, fps, power):
        """塞入每個 Epoch 結束時的精準統計數據"""
        self.epoch_indices.append(epoch)
        self.loss_history.append(loss)
        self.fps_history.append(fps)
        self.power_history.append(power)

    def draw_and_save_report(self):
        """動態繪製雙子圖並導出為實體影像檔案"""
        self.ax_loss.clear()
        self.ax_hardware.clear()

        # 🟥 左子圖：多任務綜合損失收斂曲線 (Loss Convergence)
        self.ax_loss.plot(self.epoch_indices, self.loss_history, color='#FF3366', marker='o', linewidth=2.5,
                          label='Total Loss')
        self.ax_loss.set_title("📉 Multi-Task Training Loss Convergence", fontsize=11, fontweight='bold',
                               color='#111111')
        self.ax_loss.set_xlabel("Training Epochs", fontsize=9)
        self.ax_loss.set_ylabel("MSE / BCE Multi-Head Loss", fontsize=9)
        self.ax_loss.set_xlim(1, self.total_epochs if self.total_epochs > 1 else 2)
        self.ax_loss.grid(True, linestyle='--', alpha=0.5)
        self.ax_loss.legend(loc='upper right')

        # 🟩 右子圖：訓練速度 (FPS) 與主機顯卡功耗 (Power Watts) 雙 Y 軸對比圖
        ax_fps = self.ax_hardware
        ax_pwr = self.ax_hardware.twinx()  # 建立共享 X 軸的右側 Y 軸

        # 繪製訓練速度線
        line1 = ax_fps.plot(self.epoch_indices, self.fps_history, color='#00CC66', marker='s', linestyle='--',
                            linewidth=2, label='Training Speed (FPS)')
        ax_fps.set_ylabel("Speed (Frames per Second)", color='#00CC66', fontsize=9)
        ax_fps.tick_params(axis='y', labelcolor='#00CC66')

        # 繪製顯示卡功耗線
        line2 = ax_pwr.plot(self.epoch_indices, self.power_history, color='#0066FF', marker='^', linestyle='-.',
                            linewidth=2, label='GPU Power (W)')
        ax_pwr.set_ylabel("Host GPU Power Usage (Watts)", color='#0066FF', fontsize=9)
        ax_pwr.tick_params(axis='y', labelcolor='#0066FF')

        # 整合雙 Y 軸的圖例標籤
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax_fps.legend(lines, labels, loc='lower left')

        ax_fps.set_title("⚡ Hardware Workload & Profile Progress", fontsize=11, fontweight='bold', color='#111111')
        ax_fps.set_xlabel("Training Epochs", fontsize=9)
        ax_fps.set_xlim(1, self.total_epochs if self.total_epochs > 1 else 2)
        ax_fps.grid(True, linestyle='--', alpha=0.3)

        # 導出並刷新本機磁碟影像
        self.fig.tight_layout()
        self.fig.savefig(self.report_path, dpi=150)
        print(f"📊 [Telemetry] 實時收斂與進度圖表已安全儲存至: {os.path.abspath(self.report_path)}")


# ──────────────────────────────────────────────────────────
# 2. 工廠模式：自訂多任務 2D 偵測網路與資料集定義
# ──────────────────────────────────────────────────────────
class GenericDetectionMultiTaskNet(nn.Module):
    def __init__(self, backbone_type="mobilenetv4"):
        super(GenericDetectionMultiTaskNet, self).__init__()
        self.backbone_type = backbone_type
        import timm

        if backbone_type == "mobilenetv4":
            self.backbone = timm.create_model('mobilenetv4_conv_small', pretrained=True, features_only=True)
            for p in self.backbone.parameters():
                p.requires_grad = False
            num_features = self.backbone.feature_info[-1]['num_chs']
        elif backbone_type == "ssdlite_v3":
            weights = detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            self.backbone = detection.ssdlite320_mobilenet_v3_large(weights=weights).backbone
            for p in self.backbone.parameters():
                p.requires_grad = False
            num_features = 960
        elif backbone_type == "efficientdet":
            self.backbone = timm.create_model('efficientnet_b0', pretrained=True, features_only=True)
            for p in self.backbone.parameters():
                p.requires_grad = False
            num_features = self.backbone.feature_info[-1]['num_chs']
        elif backbone_type == "mobilevit":
            self.backbone = timm.create_model('mobilevit_xxs', pretrained=True)
            for p in self.backbone.parameters():
                p.requires_grad = False
            num_features = self.backbone.num_features

        self.detect_neck = nn.Sequential(
            nn.Conv2d(num_features, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.SiLU()
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cruise_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 2))
        self.track_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 3))
        self.collision_head = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        if self.backbone_type == "mobilevit":
            features = self.backbone.forward_features(x)
        elif self.backbone_type == "ssdlite_v3":
            features = list(self.backbone(x).values())[-1]
        else:
            features = self.backbone(x)[-1]
        feat_2d = self.detect_neck(features)
        feat_1d = self.pool(feat_2d).squeeze(-1).squeeze(-1)
        return feat_2d, self.cruise_head(feat_1d), self.detection_head(feat_1d), self.collision_head(feat_1d)


class JetBotFactoryDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        if not os.path.exists(AUTO_JSON):
            os.makedirs(data_dir, exist_ok=True)
            with open(AUTO_JSON, 'w') as f:
                json.dump({"dummy.jpg": {"steering": [0.0, 0.5], "target": [0.0, 0.0, 0.0], "collision": 0}}, f)
            Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)).save(os.path.join(data_dir, "dummy.jpg"))

        with open(AUTO_JSON, 'r') as f: self.labels = json.load(f)
        self.filenames = list(self.labels.keys())

    def __len__(self): return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        img = Image.open(os.path.join(self.data_dir, name)).convert('RGB')
        d = self.labels[name]
        return (self.transform(img) if self.transform else img), \
            torch.tensor(d['steering'], dtype=torch.float32), \
            torch.tensor(d['target'], dtype=torch.float32), \
            torch.tensor(d['collision'], dtype=torch.float32)


# ──────────────────────────────────────────────────────────
# 3. 完整的兩階段微調流水線演算法 (Complete Loop Block)
# ──────────────────────────────────────────────────────────
def train_factory_pipeline(choice="mobilenetv4"):
    """
    兩階段遷移學習與知識蒸餾核心訓練演算法。
    支援自訂回歸模型 (mobilenetv4, ssdlite_v3, efficientdet, mobilevit) 與
    Ultralytics 偵測模型 (yolov11, yolo26, rf_detr)。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TOTAL_EPOCHS = 15

    # 初始化進度追蹤與收斂繪圖器
    tracker = TrainingProgressTracker(total_epochs=TOTAL_EPOCHS)

    # ──────────────────────────────────────────────────────────
    # 分支一：自訂 2D 特徵圖與多任務頭模型 (需要結合 YOLO 老師進行特徵蒸餾)
    # ──────────────────────────────────────────────────────────
    if choice in ["mobilenetv4", "ssdlite_v3", "efficientdet", "mobilevit"]:
        # 1. 初始化資料載入器
        transform = tf.Compose([
            tf.Resize((224, 224)),
            tf.ToTensor(),
            tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        dataset = JetBotFactoryDataset(DATASET_DIR, transform=transform)
        loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=False)

        # 2. 實例化學生多任務模型與優化器 (僅優化 requires_grad=True 的任務頭與頸部)
        student_model = GenericDetectionMultiTaskNet(backbone_type=choice).to(device)
        trainable_parameters = [p for p in student_model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_parameters, lr=5e-4, weight_decay=1e-4)

        # 3. 定義損失函數
        criterion_mse = nn.MSELoss()
        criterion_bce = nn.BCEWithLogitsLoss()

        # 4. 載入預訓練的 YOLOv8 老師模型，強制傳授 2D 空間定位特徵
        from ultralytics import YOLO
        teacher = YOLO("yolov8n.pt")

        print(f"🏋️ [START] 針對 【{choice.upper()}】 進行兩階段任務頭微調與特徵蒸餾...")

        # 5. 進入 Epoch 訓練主迴圈
        for epoch in range(TOTAL_EPOCHS):
            student_model.train()
            epoch_loss = 0.0
            start_time = time.time()
            images_processed = 0
            power_readings = []

            # 6. 批次資料訓練迴圈 (Batch Training Loop)
            for images, steering, target, collision in loader:
                optimizer.zero_grad()

                # 實時採集主機 NVIDIA 顯示卡當前功耗 (W)
                if HAS_NVML:
                    power = pynvml.nvmlDeviceGetPowerUsage(nvml_handle) / 1000.0
                    power_readings.append(power)
                else:
                    power_readings.append(float(np.random.uniform(150.0, 220.0)))

                # A. 執行學生模型前向傳播，取得 2D 特徵圖與 3 個任務輸出
                s_feat2d, out_steer, out_target, out_col = student_model(images.to(device))

                # B. 提取 YOLOv8 老師模型的 2D 空間特徵圖
                with torch.no_grad():
                    teacher_results = teacher.model(images.to(device))
                    t_feat2d = teacher_results

                # C. 將老師的特徵圖雙線性插值對齊學生的解析度
                t_feat_resized = nn.functional.interpolate(
                    t_feat2d,
                    size=s_feat2d.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )

                # D. 計算多頭組合損失 (巡航、跟車、防撞、特徵蒸餾)
                loss_distill = criterion_mse(s_feat2d, t_feat_resized)
                loss_task = criterion_mse(out_steer, steering.to(device)) + \
                            2.5 * criterion_mse(out_target, target.to(device)) + \
                            1.5 * criterion_bce(out_col.squeeze(1), collision.to(device))

                total_loss = loss_task + 2.0 * loss_distill

                # E. 反向傳播與優化器步伐更新
                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()
                images_processed += images.size(0)

            # 7. 每 Epoch 結束時的效能計量與實時數據統計
            elapsed = time.time() - start_time
            epoch_fps = images_processed / elapsed
            avg_loss = epoch_loss / len(loader)
            avg_power = np.mean(power_readings)

            print(
                f"Epoch {epoch + 1:02d}/{TOTAL_EPOCHS:02d} | Loss: {avg_loss:.4f} | Speed: {epoch_fps:.1f} FPS | Power: {avg_power:.1f}W")

            # 8. 同步更新歷史隊列並動態重繪 training_report.png 收斂圖表
            tracker.append_epoch_metrics(epoch=epoch + 1, loss=avg_loss, fps=epoch_fps, power=avg_power)
            tracker.draw_and_save_report()

        # 9. 儲存微調完成的權重
        save_path = f"twostage_{choice}_final.pth"
        torch.save(student_model.state_dict(), save_path)
        print(f"💾 【成功】任務頭微調權重已成功匯出至: {save_path}\n")

    # ──────────────────────────────────────────────────────────
    # 分支二：Ultralytics 家族原生端到端偵測模型 (YOLOv11, YOLO26, RT-DETR)
    # ──────────────────────────────────────────────────────────
    elif choice in ["yolov11", "yolo26", "rf_detr"]:
        from ultralytics import YOLO
        print(f"🏋️ [START] 呼叫 Ultralytics 引擎微調前沿 【{choice.upper()}】 結構...")

        # 1. 根據選擇動態載入帶有 COCO 偵權的原廠網路
        model_map = {"yolov11": "yolov11n.pt", "yolo26": "yolo26n.pt", "rf_detr": "rf-detr-n.pt"}
        if choice == "rf_detr":
            from ultralytics import RTDETR
            model = RTDETR(model_map[choice])
        else:
            model = YOLO(model_map[choice])

        # 2. 自動檢查並配置數據集路徑描述檔
        yaml_path = "jetbot_config.yaml"
        if not os.path.exists(yaml_path):
            with open(yaml_path, 'w') as f:
                f.write(f"path: ./{DATASET_DIR}\ntrain: .\nval: .\nnames:\n  0: jetbot\n  1: lane_point\n")

        # 3. 執行原生微調，並啟用 `freeze=10` 凍結前 10 層 Backbone
        start_time = time.time()
        model.train(
            data=yaml_path,
            epochs=TOTAL_EPOCHS,
            imgsz=224,
            freeze=10,
            device=0,
            verbose=False
        )
        elapsed = time.time() - start_time

        # 4. 將 Ultralytics 生成的完美收斂圖表複製成統一的報告檔案
        if os.path.exists("runs/detect/train/results.png"):
            import shutil
            shutil.copy("runs/detect/train/results.png", "training_report.png")
            print(f"📊 【成功】{choice.upper()} 官方收斂與進度圖表已安全更新至: training_report.png")

        if HAS_NVML:
            power = pynvml.nvmlDeviceGetPowerUsage(nvml_handle) / 1000.0
            print(f"💾 【成功】微調總耗時: {elapsed:.2f} 秒 | 結束時 GPU 功耗: {power:.2f} W\n")
        else:
            print(f"💾 【成功】微調總耗時: {elapsed:.2f} 秒\n")
