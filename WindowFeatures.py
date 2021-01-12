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

    # If there is less than 5 points Orientation can't be calculated and it throws an exception.
    try:
        features[0] = orientation(window, contours[0])
    except:
        features[0] = 0



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
    otherFeatures(binImage)
    print(binImage)
