import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


# 修改的 MobileNetV3 多任務模型 (PyTorch 範例)
class MobileNetV3_MultiTask(nn.Module):
    def __init__(self, backbone, num_lanes=4, num_vehicles=1):
        super().__init__()
        self.backbone = backbone
        # 車道線分割頭
        self.seg_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.Upsample(scale_factor=8, mode='bilinear'),
            nn.Conv2d(64, num_lanes, 1)
        )
        # 車輛檢測頭
        self.det_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 5 * num_vehicles)  # 假設每個車輛5個參數(x,y,w,h,conf)
        )

    def forward(self, x):
        features = self.backbone(x)
        lanes = self.seg_head(features)
        vehicles = self.det_head(features)
        return lanes, vehicles


# 初始化模型
backbone = torch.hub.load('pytorch/vision', 'mobilenet_v3_large', pretrained=True)
model = MobileNetV3_MultiTask(backbone)
model.load_state_dict(torch.load('multitask_model.pth'))
model.eval().to('cuda')

# 影像預處理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# PID 控制器
class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.reset()

    def reset(self):
        self.integral = 0
        self.prev_error = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output


# 初始化控制元件
steering_pid = PIDController(Kp=0.8, Ki=0.001, Kd=0.05)
throttle_pid = PIDController(Kp=0.5, Ki=0.001, Kd=0.05)


def process_frame(frame):
    # 車道線處理
    img = Image.fromarray(frame)
    input_tensor = transform(img).unsqueeze(0).to('cuda')

    with torch.no_grad():
        lanes, vehicles = model(input_tensor)

    # 解析車道線輸出 (假設分割輸出)
    lanes_mask = torch.argmax(lanes.squeeze(), dim=0).cpu().numpy()
    lane_center = calculate_lane_center(lanes_mask)

    # 解析車輛檢測
    vehicles = parse_vehicles(vehicles)
    safe_distance = maintain_safe_distance(vehicles)

    # 控制邏輯
    steering = steering_pid.compute(lane_center, dt=0.1)
    throttle = throttle_pid.compute(safe_distance, dt=0.1)

    return steering, throttle


def calculate_lane_center(mask):
    # 簡化版車道中心計算
    h, w = mask.shape
    left_lane = np.argmax(np.sum(mask[:, :w // 2], axis=0))
    right_lane = np.argmax(np.sum(mask[:, w // 2:], axis=0)) + w // 2
    lane_center = (left_lane + right_lane) // 2
    offset = (w // 2 - lane_center) / (w // 2)  # 歸一化偏移量
    return offset


def parse_vehicles(det_output):
    # 解析車輛邊界框 (範例格式: [x,y,w,h,conf])
    vehicles = []
    for det in det_output[0].reshape(-1, 5):
        if det[4] > 0.5:  # 置信度閾值
            vehicles.append({
                'x': det[0], 'y': det[1],
                'w': det[2], 'h': det[3]
            })
    return vehicles


def maintain_safe_distance(vehicles):
    # 安全距離邏輯 (基於邊界框大小)
    if len(vehicles) == 0:
        return 1.0  # 全速前進

    closest_vehicle = max(vehicles, key=lambda v: v['h'])
    SAFE_HEIGHT = 100  # 需根據實際校準
    if closest_vehicle['h'] > SAFE_HEIGHT:
        return -0.5  # 煞車
    else:
        return 0.2 * (1 - closest_vehicle['h'] / SAFE_HEIGHT)  # 比例減速


# 主循環 (需整合實際相機和控制介面)
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    steering, throttle = process_frame(frame)

    # 控制指令輸出 (需依硬體介面修改)
    # set_steering(steering)
    # set_throttle(throttle)

    cv2.imshow('Preview', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()