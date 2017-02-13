import numpy as np
import cv2
import glob

CAL_DIR = 'camera_cal'
CALIBRATION_IMG_WILDCARD = CAL_DIR + '/calibration*.jpg'
CAL_FILE = CAL_DIR + '/calibration_data'

OUTDIR = 'output_images'
NX = 9
NY = 6


def calibrate_camera():
    print("Calibraing camera...")

    # See if we have any saved calibration data
    mtx, dist = read_cal_data()
    if mtx is not None and dist is not None:
        print("Camera already calibrated.. returning previous results")

        img = cv2.imread('test_images/test1.jpg')
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        cv2.imwrite(OUTDIR + '/test_undist.jpg', dst)

        return mtx, dist

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

            # cv2.drawChessboardCorners(image, (NX, NY), corners, ret)
            # write_name = OUTDIR + '/corners_found' + str(idx) + '.jpg'
            # cv2.imwrite(write_name, image)

            # cv2.imshow('img', image)
            # cv2.waitKey(500)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                       imgpoints,
                                                       img_shape,
                                                       None,
                                                       None)

    img = cv2.imread('test_images/test1.jpg')
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite(OUTDIR + '/test_undist.jpg', dst)

    print("Saving calibration data")
    np.savez(CAL_FILE, mtx, dist)
    print("Done")

    return mtx, dist


def read_cal_data():
    try:
        npzfile = np.load(CAL_FILE + ".npz")
        vals = []
        for k in npzfile:
            vals.append(npzfile[k])

        return vals

    except IOError:
        print("Error loading data")
        return None, None
