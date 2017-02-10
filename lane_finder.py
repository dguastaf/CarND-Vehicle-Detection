import cv2
import numpy as np
import glob
from image_processor import ImageProcessor

CALIBRATION_IMG_WILDCARD = 'camera_cal/calibration*.jpg'
OUTDIR = 'output_images'
NX = 9
NY = 6


def calibrate_camera():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((NY * NX, 3), np.float32)
    objp[:, :2] = np.mgrid[0:NX, 0:NY].T.reshape(-1, 2)

    # 3D points in real world space
    objpoints = []

    # 2D points in image plane
    imgpoints = []

    # Save the shape of one of the images (all are the same)
    img_shape = None

    for idx, image_name in enumerate(glob.glob(CALIBRATION_IMG_WILDCARD)):
        image = cv2.imread(image_name)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if not(img_shape):
            img_shape = (image.shape[0], image.shape[1])

        ret, corners = cv2.findChessboardCorners(gray, (NX, NY), None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            cv2.drawChessboardCorners(image, (NX, NY), corners, ret)
            write_name = OUTDIR + '/corners_found' + str(idx) + '.jpg'
            cv2.imwrite(write_name, image)

            cv2.imshow('img', image)
            cv2.waitKey(500)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                       imgpoints,
                                                       img_shape,
                                                       None,
                                                       None)

    img = cv2.imread('camera_cal/calibration1.jpg')
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite(OUTDIR + '/test_undist.jpg', dst)

    return mtx, dist


mtx, dist = calibrate_camera()

processor = ImageProcessor(mtx, dist)

image = cv2.imread('test_images/straight_lines1.jpg')
processor.process_next_image(image)
