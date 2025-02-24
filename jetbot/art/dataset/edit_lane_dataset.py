import glob
import os

import cv2
import numpy as np
import pandas as pd

dir_depo = 'D:\\AI_Lecture_Demos\\Data_Repo\\Cuterbot_Repo'
dir_lane_dataset = os.path.join(dir_depo, 'lane_dataset')
dir_lane_dataset_images = os.path.join(dir_lane_dataset, 'images')
dir_lane_dataset_lane_images = os.path.join(dir_lane_dataset, 'lane_images')
dir_lane_dataset_lane_seg_data = os.path.join(dir_lane_dataset, 'lane_seg_data')

init_point = (0, 0)
lbutton_state = 0  # 左键状态标志


# https://blog.csdn.net/weixin_43794311/article/details/125802782
def mouse_click(event, x, y, flags, param=None):
    global img_path
    global img_rem
    global create_new_flag
    global img_mid
    global init_point
    global cur_point
    global lbutton_state
    global win_name
    global list_boxes

    # img_rem = cv2.imread(img_path).copy()
    # 按下左键，瞬间触发一次事件
    if event == cv2.EVENT_LBUTTONDOWN:
        if create_new_flag == 1:  # 点完右键后第一次点左键
            # 重新显示矩阵
            img_rem = cv2.imread(img_path).copy()
        else:
            img_rem = img_mid.copy()
        xl = max(min(x, 640), 0)
        yl = max(min(y, 640), 0)
        init_point = (xl, yl)  # 记录瞬间点击右键的位置
        # 记录原始点位
        # 位置信息p，和像素值信息p_v
        xy = f"{init_point}" if param else f"{init_point},{str(img_rem[x][y])}"
        cv2.circle(img_rem, (x, y), 5, (0, 255, 250), thickness=-1)
        cv2.putText(img_rem, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.5, (0, 255, 0), thickness=2)
        img_mid = img_rem.copy()  # 记录想要保存的中间状态
        create_new_flag = 0  # 不进行图片的再次刷新
        lbutton_state = 1

        # 按下左键并滑动，不松开就持续触发
        # if lbutton_state == 1 and flags == cv2.EVENT_FLAG_LBUTTON:
    if lbutton_state == 1 and flags == cv2.EVENT_FLAG_LBUTTON:
        xlc = max(min(x, 640), 0)
        ylc = max(min(y, 640), 0)

        cur_point = (xlc, ylc)
        # if not (event == cv2.EVENT_LBUTTONUP):  # 左键未松开，一直被清除
        # 重新显示矩阵
        img_rem = img_mid.copy()  # 不能直接赋值操作，会直接认为是同一地址的数据
        cv2.circle(img_rem, cur_point, 1, (0, 255, 250), thickness=1)
        # print(id(img_rem),id(img_mid))
        # cv2.imshow("mid", img_mid)

        # if lbutton_state == 1:
        cv2.rectangle(img_rem, init_point, cur_point, (0, 0, 255), 2)
        cv2.putText(img_rem, str(cur_point), (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.5, (0, 255, 0), thickness=2)

    if event == cv2.EVENT_LBUTTONUP:  # 松开左键
        lbutton_state = 0
        img_rem = img_mid.copy()
        cv2.rectangle(img_rem, init_point, cur_point, (0, 0, 255), 2)
        cv2.putText(img_rem, str(cur_point), cur_point, cv2.FONT_HERSHEY_PLAIN,
                    1.5, (0, 255, 0), thickness=2)
        img_mid = img_rem.copy()
        # cv2.imshow("mid", img_mid)
        list_boxes.append([init_point, cur_point])
        print(list_boxes)

    if event == cv2.EVENT_RBUTTONDOWN:  # 右键按下执行的动作
        if list_boxes:
            mask_out(list_boxes)
        create_new_flag = 1  # 判断是否重新打开一个图片矩阵
        img_rem = cv2.imread(img_path).copy()
        list_boxes = []

    cv2.imshow(win_name, img_rem)  # 显示的是图片矩阵


def mask_out(list_boxes):
    cv2.namedWindow('New Image', cv2.WINDOW_NORMAL)
    global img_path
    # global img_rem
    # global img_mid
    org_img_path = os.path.join(dir_lane_dataset_images, os.path.basename(img_path))
    img_new = cv2.imread(org_img_path).copy()
    cv2.imshow('New Image', img_new)
    csv_path = os.path.join(dir_lane_dataset_lane_seg_data, os.path.basename(img_path).split('.')[0] + '.csv')
    df = pd.read_csv(csv_path)
    masked_edges = df.to_numpy(dtype=np.uint8)
    mask = np.zeros(masked_edges.shape)
    mask = mask.astype('uint8')
    for b in list_boxes:
        for i in range(b[0][0], b[1][0]):
            for j in range(b[0][1], b[1][1]):
                mask[i, j] = 255
    # masked_edges *= mask
    # masked_edges_new = np.ma.masked_array(masked_edges, mask=mask, fill_value=0).copy()
    masked_edges_new = cv2.bitwise_and(masked_edges, mask)
    lines = cv2.HoughLinesP(masked_edges_new, 1, np.pi / 180, 10, minLineLength=1, maxLineGap=1)
    gray_img = cv2.cvtColor(img_rem, cv2.COLOR_BGR2GRAY)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            xm = (x2 - x1) // 2
            ym = (y2 - y1) // 2
            mask_nb = np.zeros(img_new.shape[:2], dtype="uint8")
            cv2.circle(mask_nb, (xm, ym), 5, 255, -1)
            masked_nb = cv2.bitwise_and(gray_img, gray_img, mask=mask_nb)
            ret, thresh = cv2.threshold(masked_nb, 20, 255, cv2.THRESH_BINARY_INV)
            if cv2.countNonZero(thresh) > 35:
                cv2.line(img_new, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('New Image', img_new)

    # return masked_edges
def main():
    image_paths = glob.glob(os.path.join(dir_lane_dataset_lane_images, '*.jpg'))
    global img_path
    global create_new_flag
    global lbutton_state
    global img_rem
    global img_mid
    global win_name
    global list_boxes

    # print(id(img_rem),id(img_mid))
    for img_path in image_paths:
        image = cv2.imread(img_path)
        win_name = f'image: {img_path}'
        create_new_flag = 1
        img_rem = image.copy()  # 存储一个图像矩阵
        img_mid = image.copy()  # 存储一次完成点击和松开动作的图像矩阵
        list_boxes = []
        # scale_width = 640 / image.shape[1]
        # scale_height = 480 / image.shape[0]
        # scale = min(scale_width, scale_height)
        # window_width = int(image.shape[1] * scale)
        # window_height = int(image.shape[0] * scale)
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(win_name, window_width, window_height)

        # set mouse callback function for window
        cv2.setMouseCallback(win_name, mouse_click, 1)
        cv2.imshow(win_name, img_rem)

        while True:
            key = cv2.waitKey()
            if key == 13 or key == 27 or key == ord('q'):
                break

        if key == 13:
            cv2.destroyWindow(win_name)
        elif key == 27 or key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
