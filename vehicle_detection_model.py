import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.utils import shuffle
import pickle
import os


COLORSPACE = 'HSV'
ORIENT = 9
PIX_PER_CELL = 8
CELL_PER_BLOCK = 2
HOG_CHANNEL = 'ALL'
PICKLE_FILE = "model.p"
spatial_size = (32, 32)
hist_bins = 32

KEY_SVC = "svc"
KEY_SCALER = "scaler"


def read_training_images():
    cars = glob.glob("training_images/vehicles/*/*.png")
    non_cars = glob.glob("training_images/non-vehicles/*/*.png")
    return cars, non_cars


def read_testing_images():
    cars = glob.glob("validation_images/vehicles/*/*.png")
    non_cars = glob.glob("validation_images/non-vehicles/*/*.png")
    return cars, non_cars


def save_hog_data(train_car, train_non_car, test_car, test_non_car):
    data = {}
    data["train_car"] = train_car
    data["train_non_car"] = train_non_car
    data["test_car"] = test_car
    data["test_non_car"] = test_non_car
    with open("hog.p", mode='wb') as f:
        pickle.dump(data, f)
        print("Hog data saved")


def save_model_data(svc, X_scaler):
    data = {}
    data[KEY_SVC] = svc
    data[KEY_SCALER] = X_scaler
    with open(PICKLE_FILE, mode='wb') as f:
        pickle.dump(data, f)
        print("Model data saved")


def load_hog_data():
    if os.path.exists("hog.p"):
        with open("hog.p", "rb") as f:
            data = pickle.load(f)
            return data["train_car"], data["train_non_car"], data["test_car"], data["test_non_car"]

    return None, None, None, None


def load_model_data():
    if os.path.exists(PICKLE_FILE):
        with open(PICKLE_FILE, "rb") as f:
            data = pickle.load(f)
            return data[KEY_SVC], data[KEY_SCALER]

    return None, None


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis,
                       feature_vector=feature_vec)
        return features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel],
                                            orient,
                                            pix_per_cell,
                                            cell_per_block,
                                            vis=False,
                                            feature_vec=True)

        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # Append the new feature vector to the features list
        features.append(np.hstack((spatial_features, hist_features, hog_features)))
    # Return list of feature vectors
    return features


def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32):    # bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def train_model():

    svc, X_scaler = load_model_data()
    if svc and X_scaler:
        return svc, X_scaler

    print("Training model from scratch")

    train_car_features, train_non_car_features, test_car_features, test_non_car_features = load_hog_data()

    if not train_car_features:
        cars, non_cars = read_training_images()
        test_cars, test_non_cars = read_testing_images()

        print("Extracting car features")
        train_car_features = extract_features(cars, cspace=COLORSPACE,
                                              orient=ORIENT,
                                              pix_per_cell=PIX_PER_CELL,
                                              cell_per_block=CELL_PER_BLOCK,
                                              hog_channel=HOG_CHANNEL)

        print("Extracting non-car features")
        train_non_car_features = extract_features(non_cars, cspace=COLORSPACE,
                                                 orient=ORIENT,
                                                 pix_per_cell=PIX_PER_CELL,
                                                 cell_per_block=CELL_PER_BLOCK,
                                                 hog_channel=HOG_CHANNEL)

        print("Extracting test car features")
        test_car_features = extract_features(test_cars, cspace=COLORSPACE,
                                             orient=ORIENT,
                                             pix_per_cell=PIX_PER_CELL,
                                             cell_per_block=CELL_PER_BLOCK,
                                             hog_channel=HOG_CHANNEL)

        print("Extracting test non-car features")
        test_non_car_features = extract_features(test_non_cars, cspace=COLORSPACE,
                                                 orient=ORIENT,
                                                 pix_per_cell=PIX_PER_CELL,
                                                 cell_per_block=CELL_PER_BLOCK,
                                                 hog_channel=HOG_CHANNEL)

        save_hog_data(train_car_features, train_non_car_features, test_car_features, test_non_car_features)

    train_features = train_car_features + train_non_car_features
    test_features = test_car_features + test_non_car_features

    # Create an array stack of feature vectors
    X = np.vstack((train_features, test_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    X_train = scaled_X[0:len(train_features)]
    X_test = scaled_X[len(train_features):]

    X_train_car = X_train[0:len(train_car_features)]
    assert(len(X_train_car) == len(train_car_features))

    X_train_non_car = X_train[len(train_car_features):]
    assert(len(X_train_non_car) == len(train_non_car_features))

    X_test_car = X_test[0:len(test_car_features)]
    assert(len(X_test_car) == len(test_car_features))
    X_test_non_car = X_test[len(test_car_features):]
    assert(len(X_test_non_car) == len(test_non_car_features))

    y_train = np.hstack((np.ones(len(X_train_car)), np.zeros(len(X_train_non_car))))
    y_test = np.hstack((np.ones(len(X_test_car)), np.zeros(len(X_test_non_car))))

    assert(len(X_test) == len(test_car_features + test_non_car_features))
    assert(len(X_train) == len(train_car_features + train_non_car_features))

    print("Training size: {} Test size: {}".format(len(X_train), len(X_test)))

    print('Using:', ORIENT, 'orientations', PIX_PER_CELL,
          'pixels per cell and', CELL_PER_BLOCK, 'cells per block')
    print('Feature vector length:', len(X_train[0]))

    # Use a linear SVC
    svc = LinearSVC()

    X_train, y_train = shuffle(X_train, y_train)

    svc.fit(X_train, y_train)
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    n_predict = 100

    X_test, y_test = shuffle(X_test, y_test)

    print('My SVC predicts:     ', svc.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])

    save_model_data(svc, X_scaler)

    return svc, X_scaler
