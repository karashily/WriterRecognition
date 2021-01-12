import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

def angle_between(p1, p2):
    ang = np.arctan2(p2[:,1] - p1[:,1], p2[:,0] - p1[:,0])
    return np.rad2deg(ang % (2 * np.pi))

def length(p1,p2):
    return np.linalg.norm(p1-p2, axis=1)

def vectorLength(v):
    return np.linalg.norm(v, axis=1)

def dot(v1,v2):
    return np.dot(v1, v2)

def vector(p1, p2):
    return p2 - p1

def PolygonFeatures(image_name):    
    
    image = cv2.imread(image_name) 
    
    # Grayscale 
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    
    # Find Canny edges 
    
    edged = cv2.Canny(image, 30, 200) 
    
    # Finding Contours 
    # Use a copy of the image e.g. edged.copy() 
    # since findContours alters the image 
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    
    #print("Number of Contours found = " + str(len(contours))) 

    # Draw all contours 
    # -1 signifies drawing all contours 
    #cv2.drawContours(image, contours, -1, (0, 255, 0), 3) 
    #cv2.imshow('Contours', image) 
    
    dividand = 20
    buckets = 180//dividand

    f10 = np.zeros(buckets)
    f11 = np.zeros(buckets)
    f12 = np.zeros(buckets)
    f13 = np.zeros(buckets)
    lengths = []

    total_length1 = 0
    total_length2 = 0
    
    for i in range(len(contours)):
        contours[i] = contours[i].squeeze()
        if(len(contours[i].shape)==1 or contours[i].shape[0] < 2):
            continue
            
        p1, p2 = contours[i][:-1], contours[i][1:]  #points
        
        #f10, f11, f14
        slopes = angle_between(p1, p2)              #slope of lines
        segmentsLength = length(p1, p2)             #length of lines
        largerThan180 = slopes > 180                #positions of slopes > 180
        slopes[largerThan180] -= 180                
        slopes -= 0.1                               #margin to make the values fit in the buckets
        negativeSlopes = slopes < 0                 #positions of negative slopes due to margin
        slopes[negativeSlopes] = 0                  #make the negative slopes = 0
        Buckets = (slopes/dividand).astype(int)     #add each slope to its bucket
        
        for i in range(Buckets.shape[0]):           #modify histograms
            f10[Buckets[i]] += 1
            f11[Buckets[i]] += segmentsLength[i]
            total_length1 += segmentsLength[i]
            lengths.append(segmentsLength[i])
        
        #f12, f13
        vectors = vector(p1, p2)                                        #generate all vectors
        vectorsLengths = vectorLength(vectors)                          #get length of all vectors
        v1, v2 = vectors[:-1], vectors[1:]                              #line up vectors for a vectorized code
        len1, len2 = vectorsLengths[:-1], vectorsLengths[1:]            #line up lengthes for a vectorized code
        nominator = np.sum(v1 * v2, axis = 1)                           #apply dot product between each corresponding vectors
        denominator = len1 * len2                                       #apply length multiplication between each corresponding vectors
        angles = np.rad2deg(np.pi - np.arccos(nominator / denominator)) #calculate the angle between each corresponding vectors
        angles -= 0.1                                                   #margin to make the values fit in the buckets
        negativeSlopes = angles < 0                                     #positions of negative angles due to margin
        angles[negativeSlopes] = 0                                      #make the negative angles = 0
        Buckets = (angles/dividand).astype(int)                         #add each angle to its bucket
        for i in range(Buckets.shape[0]):                               #modify histograms
            f12[Buckets[i]] += 1
            f13[Buckets[i]] += len1[i] + len2[i]
            total_length2 += len1[i] + len2[i]
            
    f11 /= total_length1
    f13 /= total_length2
    f14, bin_edges = np.histogram(lengths,buckets * 2,density = True)
    
    return f10, f11, f12, f13, f14
