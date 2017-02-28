import numpy as np
import cv2
from vehicle_detection_model import train_model, get_hog_features

orient = 9
window = 64
pix_per_cell = 8
cell_per_block = 2
ystart = 390
ystop = 656


def draw_rect(img, xleft, ytop, scale):
    xbox_left = np.int(xleft * scale)
    ytop_draw = np.int(ytop * scale)
    win_draw = np.int(window * scale)
    cv2.rectangle(img, (xbox_left, ytop_draw + ystart),
                  (xbox_left + win_draw,
                   ytop_draw + win_draw + ystart),
                  (0, 0, 255), 6)


# Define a single function that can extract features using hog sub-sampling
# and make predictions
def find_cars_with_scale(img, svc, X_scaler, scale):

    img = np.copy(img)

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch,
                                     (np.int(imshape[1] / scale),
                                      np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - 1

    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 3  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell,
                            cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell,
                            cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell,
                            cell_per_block, feature_vec=False)

    found_boxes = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window,
                             xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window,
                             xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window,
                             xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Scale features and make a prediction
            test_features = X_scaler.transform(hog_features)
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                found_boxes.append((xleft, ytop, scale))

    return found_boxes


def find_cars(img):
    svc, X_scaler = train_model()
    scales = [.6, 1.2, 1.75]

    found_boxes = []
    for scale in scales:
        found_boxes = found_boxes + \
            find_cars_with_scale(img, svc, X_scaler, scale)

    out_img = np.copy(img)
    for xleft, ytop, scale in found_boxes:
        draw_rect(out_img, xleft, ytop, scale)
    return out_img
