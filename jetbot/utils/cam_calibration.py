import os.path
import time

import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# gst_str = "nvarguscamerasrc sensor_mode=2 ! nvvidconv flip-method=0 ! 'video/x-raw,width=960, height=616' ! nvvidconv ! ximagesink"
gst_str = 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=640, format=(string)NV12, framerate=30/1 ! nvvidconv flip-method=0 ! video/x-raw, width=1640, height=1232, format=(string)BGRx ! videoconvert ! appsink'
cap = cv.VideoCapture(gst_str, cv.CAP_GSTREAMER)

# dir_patterns = "D:\\AI_Lecture_Demos\\Data_Repo\\Cuterbot_Repo\\snapshots"
# test_images = os.path.join(dir_patterns, 'checkerboard pattern 1.jpg')
# dir_images = "D:\\AI_Lecture_Demos\\Data_Repo\\Cuterbot_Repo\\lane_dataset\\images"
# test_images = "/home/cuterbot/Cuterbot_Demo/notebooks/teleoperation/snapshots/ecb7fa78-f655-11ef-a9e5-7cb27d304b9d.jpg"
# test_img = cv.imread(test_images)
# for fname in images:
while True:
    # img = cv.imread(fname)
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # while True:
    if not cap.isOpened():
        cap.open(gst_str, cv.CAP_GSTREAMER)
    re, cap_image = cap.read()
    # if key == 13:
    #     gray = cv.cvtColor(cap_image, cv.COLOR_BGR2GRAY)
    #     cv.destroyWindow('cap_img')
    #    break
    # cv.destroyWindow('cap_img')
    # time.sleep(1)
    gray = cv.cvtColor(cap_image, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7, 6), None)
    # print(re, ret)
    cv.namedWindow('cap_img', cv.WINDOW_NORMAL)
    cv.resizeWindow('cap_img', 960, 640)
    cv.imshow('cap_img', cap_image)
    cv.waitKey(500)
    # If found, add object points, image points (after refining them)
    if ret is True:
        cv.destroyWindow('cap_img')
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(cap_image, (7, 6), corners2, ret)
        cv.namedWindow('img', cv.WINDOW_NORMAL)
        cv.resizeWindow('img', 960, 640)
        cv.imshow('img', cap_image)
        key = cv.waitKey(0)
        if key == 13:
            break
cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

test_images = "/home/cuterbot/Cuterbot_Demo/notebooks/teleoperation/snapshots/e93b81ee-f655-11ef-a9e5-7cb27d304b9d.jpg"
# test_images = "/home/cuterbot/Cuterbot_Demo/notebooks/teleoperation/snapshots/xy_000_334_e4e72c2a-c279-11ef-82a3-7cb27d304b9d.jpg"
test_img = cv.imread(test_images)

h, w = test_img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
# undistort
dst = cv.undistort(test_img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
# cv.imwrite('calibresult.png', dst)
cv.namedWindow('undist_img', cv.WINDOW_NORMAL)
cv.resizeWindow('undist_img', 960, 640)
cv.imshow('undist_img', dst)
cv.waitKey(0)
