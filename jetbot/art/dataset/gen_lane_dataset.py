import glob
from zipfile import ZipFile

import cv2
import numpy as np
import os
import pandas as pd

def detect_lanes(frame):
    # 轉換為灰度圖
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray_img = frame
    # 高斯模糊
    # blur_img = cv2.GaussianBlur(gray_img, (7, 7), sigmaX=1.5, sigmaY=1.5)
    blur_img = cv2.medianBlur(gray_img, 9)

    # Canny 邊緣檢測
    # edges = cv2.Canny(blur_img, 50, 150)
    edges = cv2.Canny(blur_img, 50, 85, L2gradient=True, apertureSize=3)

    # 創建遮罩
    mask = np.zeros_like(edges)
    height, width = edges.shape
    # polygon = np.array([[(0, height), (width // 2, height // 2), (width, height)]], np.int32)
    polygon = np.array([[(0, height), (0, height // 2.0), (width // 8, height // 3),
                         (width // 1.5, height // 2.5), (width, height)]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # 霍夫變換檢測直線
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 10, minLineLength=1, maxLineGap=1)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            xm = (x2 - x1) // 2
            ym = (y2 - y1) // 2
            mask_nb = np.zeros(frame.shape[:2], dtype="uint8")
            cv2.circle(mask_nb, (xm, ym), 5, 255, -1)
            masked_nb = cv2.bitwise_and(gray_img, gray_img, mask=mask_nb)
            ret, thresh = cv2.threshold(masked_nb, 20, 255, cv2.THRESH_BINARY_INV)
            if cv2.countNonZero(thresh) > 35:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame, masked_edges


dir_depo = 'D:\\AI_Lecture_Demos\\Data_Repo\\Cuterbot_Repo'
os.makedirs(dir_depo, exist_ok=True)

training_datafile = 'road_following_dataset_xy_2024-12-25_04-45-04.zip'  # check the data file is loaded to dir_depo
dir_lane_dataset = os.path.join(dir_depo, 'lane_dataset')
os.makedirs(dir_lane_dataset, exist_ok=True)
dir_lane_dataset_images = os.path.join(dir_lane_dataset, 'images')
os.makedirs(dir_lane_dataset_images, exist_ok=True)
dir_lane_dataset_lane_images = os.path.join(dir_lane_dataset, 'lane_images')
os.makedirs(dir_lane_dataset_lane_images, exist_ok=True)
dir_lane_dataset_lane_seg_data = os.path.join(dir_lane_dataset, 'lane_seg_data')
os.makedirs(dir_lane_dataset_lane_seg_data, exist_ok=True)

with ZipFile(os.path.join(dir_depo, training_datafile), 'r') as zObject:
    file_list = zObject.namelist()
    for zip_info in zObject.infolist():
        if zip_info.is_dir():
            continue
        zip_info.filename = os.path.basename(zip_info.filename)
        zObject.extract(zip_info, dir_lane_dataset_images)


def main():
    image_paths = glob.glob(os.path.join(dir_lane_dataset_images, '*.jpg'))
    for img in image_paths:
        image = cv2.imread(img)
        lane_frame, masked_edges = detect_lanes(image)
        df = pd.DataFrame(masked_edges)
        df.to_csv(path_or_buf=os.path.join(dir_lane_dataset_lane_seg_data,
                                           os.path.basename(img).split('.')[0]+'.csv'), index=False)
        # color_edges = np.dstack(masked_edges, masked_edges, masked_edges)
        cv2.imwrite(os.path.join(dir_lane_dataset_lane_images,
                                 os.path.basename(img)), lane_frame)
        # cv2.imshow("Lane Detection", lane_frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
