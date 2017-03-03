import numpy as np
import cv2
from vehicle_detection_model import *
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
import matplotlib.image as mpimg

orient = 9
window = 64
pix_per_cell = 8
cell_per_block = 2
ystart = 400
ystop = 656
cells_per_step = 2


def draw_labels(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                (np.max(nonzerox), np.max(nonzeroy)))

        if (bbox[1][0] - bbox[0][0] > 20 and bbox[1][1] - bbox[0][1] > 20):
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


def draw_rect(img, boxes):
    for box in boxes:
        x1 = box[0][0]
        y1 = box[0][1]
        x2 = box[1][0]
        y2 = box[1][1]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 6)


# Define a single function that can extract features using hog sub-sampling
# and make predictions
def find_cars_with_scale(img, svc, X_scaler, scale):
    img = img.astype(np.float32)/255
    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
    # ctrans_tosearch = ctrans_tosearch.astype(np.float32)/255

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
      # Instead of overlap, define how many cells to step
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

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                x1 = xbox_left
                y1 = ytop_draw + ystart
                x2 = xbox_left + win_draw
                y2 = ytop_draw + win_draw + ystart

                found_boxes.append(((x1, y1), (x2, y2)))

    return found_boxes


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def find_cars(img):
    svc, X_scaler = train_model()
    scales = [.9, 1.5, 2]

    heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    all_boxes = []
    for scale in scales:
        found_boxes = find_cars_with_scale(img, svc, X_scaler, scale)
        all_boxes += found_boxes
        heat = add_heat(heat, found_boxes)

    thresh_heat = apply_threshold(heat, 2)
    clip_heat = np.clip(thresh_heat, 0, 255)
    labels = label(clip_heat)
    # plt.imshow(labels[0], cmap='gray')
    # plt.title('Heat Map')
    # plt.show()

    out_img = np.copy(img)
    draw_rect(out_img, all_boxes)
    # draw_labels(out_img, labels)
    return out_img
