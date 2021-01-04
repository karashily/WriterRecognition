import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

def angle_between(p1, p2):
    ang = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    return np.rad2deg(ang % (2 * np.pi))

def length(p1,p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def dot(v1,v2):
    return v1[0] * v2[0] + v1[1] * v2[1]

def vector(p1, p2):
    return np.array([p2[0] - p1[0], p2[1] - p1[1]])

def PolygonFetures(image_name):
    image = cv2.imread(image_name) 
    cv2.waitKey(0) 
    
    # Grayscale 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    
    # Find Canny edges 
    edged = cv2.Canny(gray, 30, 200) 
    
    # Finding Contours 
    # Use a copy of the image e.g. edged.copy() 
    # since findContours alters the image 
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    
    print("Number of Contours found = " + str(len(contours))) 

    # Draw all contours 
    # -1 signifies drawing all contours 
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3) 
    cv2.imshow('Contours', image) 
    cv2.waitKey(0) 

    dividand = 20
    buckets = 180//dividand

    f10 = np.zeros(buckets)
    f11 = np.zeros(buckets)
    f12 = np.zeros(buckets)
    f13 = np.zeros(buckets)
    lengthes = []

    total_length1 = 0
    total_length2 = 0
    
    for i in range(len(contours)):
        contours[i] = contours[i].squeeze()
        print("contour: ",i + 1, contours[i].shape)
        for j in range(1,contours[i].shape[0]):
            p1, p2 = contours[i][j-1], contours[i][j]

            angle = angle_between(p1, p2)
            if angle > 180:
                angle -= 180

            bucket = int(np.max(angle-0.1,0)/dividand)
            
            f10[bucket] += 1

            dst = length(p1, p2)
            f11[bucket] += dst
            total_length1 += dst

            lengthes.append(dst)

        for j in range(2,contours[i].shape[0]):
            p1, p2, p3 = contours[i][j-2], contours[i][j-1], contours[i][j]
            v1, v2 = vector(p1,p2), vector(p2,p3)
            len1, len2 = length(p1,p2), length(p2,p3)
            angle = np.rad2deg(np.pi - np.arccos(dot(v1, v2) / (len1 * len2)))

            bucket = int(np.max(angle-0.1,0)/dividand)

            f12[bucket] += 1
            f13[bucket] += len1 + len2
            total_length2 += len1 + len2

    f11 /= total_length1
    f13 /= total_length2
    f14, bin_edges = np.histogram(lengthes,buckets * 2,density = True)

    return f10, f11, f12, f13, f14