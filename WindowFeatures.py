import cv2 
import numpy as np
from sklearn.preprocessing import normalize as norm


def horHist(window):
    """
    It computes horizontal histogram of the window
    :param window: NxN codeword window[0-255]
    :return: horizontal histogram
    """
    window = window / 255
    proj = np.sum(window, axis=1)
    return proj


def vertHist(window):
    """
    It computes vertical histogram of the window
    :param window: NxN codeword window[0-255]
    :return: vertical histogram
    """
    window = window / 255
    proj = np.sum(window, axis=0)
    return proj


def profiles(windows):
    """
    Upper/lower profiles are computed by considering, for each image column, the distance between
    the horizontal line y=yt  and the closest pixel to the upper/lower boundary of the character image

    yt is the average y that have pixel values > 0

    :param window: NxN codeword window [0-255]
    :return: Upper & Lower profiles
    """
    window = windows / 255
    yt = np.sum(np.transpose((np.transpose(window) * np.arange(0, window.shape[0])))) / np.sum(window)
    yt = int(np.round(yt))
    up_profile = np.zeros(window.shape[0])
    low_profile = np.zeros(window.shape[0])

    for colIdx, col in enumerate(np.transpose(window)):

        # Upper Profile:
        fst_1 = np.argmax(col)
        if fst_1 < yt and col[fst_1] > 0:
            up_profile[colIdx] = yt - fst_1

        # Lower Profile:
        lst_1 = np.argmax(col[::-1])
        if lst_1 < yt and col[fst_1] > 0:
            low_profile[colIdx] = window.shape[1] - lst_1 - 1 - yt
        """
        for idx in range(0, yt+1):
            if col[idx] > 0:
                up_profile[colIdx] = yt - idx
                break

        # Lower Profile:
        for idx in range(yt, window.shape[1]):
            if col[idx] > 0:
                low_profile[colIdx] = idx - yt
        """
    return up_profile, low_profile


def otherFeatures(window):
    """
    Calculates Orientation, Eccentricity, Rectangularity, Elongation, Perimeter, Solidity features

    :param window: NxN codeword window [0-255]
    :return: numpy array of 6 features
    """
    features = np.zeros(6)
    contours, _ = cv2.findContours(window, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Calculating Orientation & Eccentricity from the fitting ellipse:
    # Orientation -> it is the angle at which object is directed
    # Eccentricity -> it is square root of (1 - (ratio between axes of the enclosing ellipse)^2)

    # If there is less than 5 points Ellipse can't be created, therefore orientation and eccentricity cant be calculated
    try:
        center, axes, angle = cv2.fitEllipse(contours[0])
        features[0] = angle
        features[1] = np.sqrt(1-(min(axes)/max(axes))**2)
    except:
        features[0] = 0
        features[1] = 0

    # Calculating Rectangularity:
    # Rectangularity -> ratio of the region area to the minimum bounding rectangle
    try:
        rect = cv2.minAreaRect(contours[0])
    # print(rect)
        features[2] = cv2.countNonZero(window) / (rect[1][0] * rect[1][1])
    except:
        features[2] = 0

    # Calculating Elongation:
    # Elongation is calculated as stated in equation below
    try:
        m = cv2.moments(window)
        x = m['mu20'] + m['mu02']
        y = 4 * m['mu11'] ** 2 + (m['mu20'] - m['mu02']) ** 2
        features[3] = (x + y ** 0.5) / (x - y ** 0.5)
    except:
        features[3] = 0
        
    # Calculating Perimeter:
    try:
        features[4] = cv2.arcLength(contours[0], True)
    except:
        features[4] = 0
        
    # Calculating Solidity:
    # Solidity -> It is the ratio of the contour area to the convex hull area
    try:
        features[5] = cv2.contourArea(contours[0]) / cv2.contourArea(cv2.convexHull(contours[0]))
    except:
        features[5] = 0
    
    return features


def WindowFeatures(window_):
    
    #image = cv2.cvtColor(window_, cv2.COLOR_BGR2GRAY)
    _, binImage = cv2.threshold(window_, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    window = 255 - binImage
    
    n = window.shape[0]
    features = np.zeros(4*n + 6)

    features[:n] = horHist(window)
    features[n: 2*n] = vertHist(window)

    upper_profile, lower_profile = profiles(window)
    features[2*n: 3*n] = upper_profile
    features[3*n: 4*n] = lower_profile

    features[4*n:] = otherFeatures(window)

    normalized = norm(features[:, np.newaxis], axis=0).ravel()

    return normalized


if __name__ == "__main__":
    image = cv2.imread("test.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    temp = image[83:96, 72:85]
    _, binImage = cv2.threshold(temp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    binImage = 255 - binImage

    binImage[-1] = np.zeros(13)
    binImage[-1, 12] = 255
    binImage[-2, :9] = 0
    binImage[-3, :9] = 0
    binImage[-4, :7] = 0

    print(WindowFeatures(binImage))
