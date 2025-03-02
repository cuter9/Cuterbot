import numpy as np
import cv2
import glob
import os

''' References : 
    https://gist.github.com/mesutpiskin/0ced27981487491403610324fea55038
    https://shichaoxin.com/2024/08/08/%E9%B1%BC%E7%9C%BC%E7%9B%B8%E6%9C%BA-Fisheye-camera-model-in-OpenCV/
    https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0
    https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-part-2-13990f1b157f
    https://blog.csdn.net/qq_39642978/article/details/112742933
    https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d
    https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html
'''
# Define the chess board rows and columns
CHECKERBOARD = (6, 9)
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# dir_patterns = "/home/cuterbot/Cuterbot_Demo/notebooks/teleoperation/snapshots"
dir_patterns = "D:\AI_Lecture_Demos\Data_Repo\Cuterbot_Repo\snapshots"
images = os.path.join(dir_patterns, '*.jpg')

counter = 0
for path in glob.glob(images):
    # Load the image and convert it to gray scale
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    # Make sure the chess board pattern was found in the image
    if ret:
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, (7, 6), corners, ret)
    print(ret, str(path))
    counter += 1

N_imm = counter  # number of calibration images
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_imm)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_imm)]
rms, _, _, _, _ = cv2.fisheye.calibrate(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    K,
    D,
    rvecs,
    tvecs,
    calibration_flags,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

''' 焦距比例控制，控制视场大小
    https://blog.csdn.net/guanjing_dream/article/details/133576214?spm=1001.2014.3001.5502
    https://tencentcloud.csdn.net/67888801edd0904849a56cf4.html?dp_token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpZCI6MjQ3Nzc3MywiZXhwIjoxNzQxNDIzMDM2LCJpYXQiOjE3NDA4MTgyMzYsInVzZXJuYW1lIjoiY3V0ZXI5In0.BrzEl5uyk_Piwtd93gFn0qYMpD5DJ3bEOqjOmf3BXx4&spm=1001.2101.3001.6650.8&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-8-134042387-blog-133576214.235%5Ev43%5Epc_blog_bottom_relevance_base5&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-8-134042387-blog-133576214.235%5Ev43%5Epc_blog_bottom_relevance_base5&utm_relevant_index=12
'''
fs = 1.5
newK = K.copy()
newK[0][2] = fs * newK[0][2]
newK[1][2] = fs * newK[1][2]

dir_lane_img = 'D:\AI_Lecture_Demos\Data_Repo\Cuterbot_Repo\lane_dataset\images'
test_image_name = "xy_180_319_592f55da-c27a-11ef-82a3-7cb27d304b9d.jpg"
# test_image = os.path.join(dir_lane_img, test_image_name)
test_image = os.path.join(dir_lane_img, "*.jpg")


for tm in glob.glob(test_image):
    test_img_org = cv2.imread(tm)
    border_x, border_y = int(test_img_org.shape[0]/4), int(test_img_org.shape[1]/4)
    width, height = (int(test_img_org.shape[0] + test_img_org.shape[0]/2),
                     int(test_img_org.shape[1] + test_img_org.shape[1]/2))
    test_img = cv2.copyMakeBorder(test_img_org, border_y, border_y, border_x, border_x, cv2.BORDER_ISOLATED)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(newK, D, np.eye(3), newK, (width, height), cv2.CV_16SC2)
    undistorted_img = cv2.remap(test_img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Original Image', test_img)
    cv2.namedWindow('Undistort Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Undistort Image', undistorted_img)
    key = cv2.waitKey(0)
    if key == 27 or key == ord('q'):
        break
    else:
        continue
