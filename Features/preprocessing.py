import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import time

def cutHandWriting(img):
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(img, kernel, iterations=4)

    invert = np.invert(erosion)
    _, thr = cv2.threshold(invert, 0, 255, cv2.THRESH_OTSU)

    rho = 1                          # distance resolution in pixels of the Hough grid
    theta = np.pi / 180              # angular resolution in radians of the Hough grid
    threshold = 300                  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 1800           # minimum number of pixels making up a line
    max_line_gap = 10                # maximum gap in pixels between connectable line segments
    line_image = np.zeros_like(img)  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(thr, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

    rows = np.sum(line_image, axis=1)
    peaks, _ = find_peaks(rows, height=10, distance=50)


    if len(peaks) < 2 or peaks[-2]-peaks[-1] < 1700:
        min_line_length = 1700
        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(thr, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

        rows = np.sum(line_image, axis=1)
        peaks, _ = find_peaks(rows, height=10, distance=50)

        if len(peaks) < 2:
            return img


    extracted_img = img[peaks[-2]:]
    

    return extracted_img