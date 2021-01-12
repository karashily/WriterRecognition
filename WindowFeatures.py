import cv2 
import numpy as np


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


def profiles(window):
    """
    Upper/lower profiles are computed by considering, for each image column, the distance between
    the horizontal line y=yt  and the closest pixel to the upper/lower boundary of the character image

    yt is the average y that have pixel values > 0

    :param window: NxN codeword window [0-255]
    :return: Upper & Lower profiles
    """
    window = window / 255
    yt = np.sum(np.transpose((np.transpose(binImage) * np.arange(0, window.shape[0])))) / np.sum(window)
    yt = int(np.round(yt))

    up_profile = np.zeros(window.shape[0])
    low_profile = np.zeros(window.shape[0])

    for colIdx, col in enumerate(np.transpose(window)):

        # Upper Profile:
        for idx in range(0, yt+1):
            if col[idx] > 0:
                up_profile[colIdx] = yt - idx
                break

        # Lower Profile:
        for idx in range(yt, window.shape[1]):
            if col[idx] > 0:
                low_profile[colIdx] = idx - yt

    return up_profile, low_profile


def otherFeatures(window):

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
    rect = cv2.minAreaRect(contours[0])
    features[2] = cv2.countNonZero(window) / (rect[1][0] * rect[1][1])

    # Calculating Elongation:
    # Elongation is calculated as stated in equation below
    m = cv2.moments(window)
    x = m['mu20'] + m['mu02']
    y = 4 * m['mu11'] ** 2 + (m['mu20'] - m['mu02']) ** 2
    features[3] = (x + y ** 0.5) / (x - y ** 0.5)

    # Calculating Perimeter:
    features[4] = cv2.arcLength(contours[0], True)
    print(features[4])


def orientation(window, contours):
    """
    It computes the angle at which the object is directed

    :param window: NxN codeword window
    :param contours: contours of the writings in the window
    :return:
    """
    center, axes, angle = cv2.fitEllipse(contours)
    return angle


def eccentricity(window):
    pass


def rectangularity(window):
    pass


def elongation(window):
    pass


def perimeter(window):
    pass


def solidity(window):
    pass


def WindowFeatures(window):

    window = np.zeros((13, 13))
    n = window.shape[0]
    features = np.zeros(4*n + 6)

    # contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    return 0


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
    otherFeatures(binImage)
    print(cv2.countNonZero(binImage))
