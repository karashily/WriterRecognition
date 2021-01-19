import skimage.io as io
import time
import numpy as np
import cv2
import os
from sklearn.svm import SVC
from Features.FeatureExtractor import *

test_cases = os.listdir("TestData")
test_cases.sort()

time_file = open("time.txt", "w")
results_file = open("results.txt", "w")
for case in test_cases:
    
    writer1_1  = cv2.imread('TestData/'+case+"/1/1.png", cv2.IMREAD_GRAYSCALE)
    writer1_2  = cv2.imread('TestData/'+case+"/1/2.png", cv2.IMREAD_GRAYSCALE)
    writer2_1  = cv2.imread('TestData/'+case+"/2/1.png", cv2.IMREAD_GRAYSCALE)
    writer2_2  = cv2.imread('TestData/'+case+"/2/2.png", cv2.IMREAD_GRAYSCALE)
    writer3_1  = cv2.imread('TestData/'+case+"/3/1.png", cv2.IMREAD_GRAYSCALE)
    writer3_2  = cv2.imread('TestData/'+case+"/3/2.png", cv2.IMREAD_GRAYSCALE)
    test_img =  cv2.imread('TestData/'+case+"/test.png", cv2.IMREAD_GRAYSCALE)
    print("Processing Test Case : " , case)

    st = time.time()
  
    
    
    f1_1  = Feature_Extractor(writer1_1)
    f1_2  = Feature_Extractor(writer1_2)
    f2_1  = Feature_Extractor(writer2_1)
    f2_2 = Feature_Extractor(writer2_2)
    f3_1  = Feature_Extractor(writer3_1)
    f3_2  = Feature_Extractor(writer3_2)
    pred  = Feature_Extractor(test_img)
    
    features = np.array([f1_1,f1_2,f2_1,f2_2,f3_1,f3_2])
 
    
    
    classifier = SVC()
    classifier.fit(features, [1,1,2,2,3,3])
    result = classifier.predict([pred])
    
    
    en = time.time()
    time_file.write(str(round((en-st),2))+'\n')
    results_file.write(str(result[0])+'\n')
time_file.close()
results_file.close()

    
    
    
    
