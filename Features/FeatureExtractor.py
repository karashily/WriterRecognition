import skimage.io as io
import time
import numpy as np
import cv2
import os

from .preprocessing import *
from .LocalBinaryPattern import *
from .ChainCodesFeatures import *
from .PolygonFeatures import * 
from .WindowFeatures import *
from .windows import *

def Feature_Extractor(img):
    
    
    
    #img = cutHandWriting(img)
    
    img = cv2.resize(img,(img.shape[0]//4,img.shape[1]//4))
    
    #lbp = LocalBinaryPatterns(24,8)
    #lbp_features = lbp.describe(img)
    #st = time.time()
    edged = cv2.Canny(img, 30, 200)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #en = time.time()
#     print("Contours : ",(en-st)*1000,"ms")
    
    #st = time.time()
    #windows, clusters, clusteredWindows = getClusteredWindows(img,13,10)
    #en = time.time()
#     print("Kesho : ",(en-st)*1000,"ms")
    
    #st = time.time()
    F5 = Chain_Codes_Feature_Extractor(img,contours,None)
    #en = time.time()
#     print("Ramzy : ",(en-st)*1000,"ms")
    
    #st = time.time()
    #f10, f11, f12, f13, f14 = PolygonFeatures(contours)
    #en = time.time()
#     print("Ibrahim Lefta : ",(en-st)*1000,"ms")
    
    #st = time.time()
    #Windows = list()
    
    #for i in range(clusteredWindows.shape[0]):
        #print(clusteredWindows[i])
        #Windows  += list(WindowFeatures(clusteredWindows[i].astype(np.uint8)))
    #en = time.time()
#     print("October : ",(en-st)*1000,"ms")
    return F5
    
    
    