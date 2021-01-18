import skimage.io as io
import numpy as np
from skimage.filters import threshold_otsu 
from skimage.color import rgb2grey
from skimage.measure import find_contours
import cv2



def First_difference_VEC(chain_code):
    #chain =chain_code[-1] +chain_code

    chain_shifted = chain_code[1:]
    chain = chain_code[:-1]
    first_diff = -1*(chain - chain_shifted)%8
    
    return first_diff


def Chain_Codes_Counting_VEC(contours,pairs_triplets=True,second_order=False):
    directions = [[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1],[0,1],[1,1]]
    global_chain_code =list()#= np.array([])
    first_difference = list()#np.array([])
    second_difference = list()#np.array([])
    p_t = list()
    pairs_tri = np.array([])
    for cont in contours:
        cont = np.squeeze(cont,1)
        cont_diff= cont [1:]-cont[:-1]
        chain_code = np.zeros(len(cont_diff),dtype=np.int)
        for i in range(len(directions)):
            t = (cont_diff==directions[i])
            chain_code[t[:,0] & t[:,1]]=i
        if(len(chain_code)>0 and second_order):
            first_diff = First_difference_VEC(chain_code)
            first_difference += list(first_diff)#= np.append(first_diff,first_difference)    
            second_difference += list(First_difference_VEC(first_diff)) #= np.append(First_difference_VEC(first_diff),second_difference)
        global_chain_code+=list(chain_code) #= np.append(chain_code,global_chain_code)
        if(pairs_triplets):
            p_t += list(chain_code)
            p_t += [9]#To Divide between two chain codes so i can count the pairs , triplets directly from string
                      #Must be int and not negative and not in range(0--7) because it will change count , -1 = "-1"

    if(pairs_triplets):
        pairs_tri = np.array(p_t,dtype=np.str)
        pairs_tri = ''.join(pairs_tri)
        f4 = np.zeros(64)
        f5 = np.zeros(512)
        for  i in range(8):
            for j in range(8):
                check = str(i)+str(j)
                f4[i*8+j] = pairs_tri.count(check)
                for k in range(8):
                    check = str(i)+str(j)+str(k)
                    f5[i*64+j*8+k] = pairs_tri.count(check)



    if(second_order):
        f1,_ = np.histogram(global_chain_code,8)
        f2,_ = np.histogram(first_difference,8)
        f3,_ = np.histogram(second_difference,8)
        f2 = np.append(f2[:5],f2[6:])
        if(np.sum(f1)!=0):
            f1 = f1/np.sum(f1)
        if(np.sum(f2)!=0):
            f2 = f2/np.sum(f2)
        if(np.sum(f3)!=0):
            f3 = f3/np.sum(f3)
    
    if pairs_triplets:
        if(np.sum(f4)!=0):
            f4 = f4/f4.sum()
        if(np.sum(f5)!=0):
            f5 = f5/f5.sum()
        return f5
        return f1,f2,f3,f4,f5
    
    
    return f1,f2,f3


def Chain_Codes_Feature_Extractor(img,contours,clusteredWindows):
    #img = io.imread(image_name)
    #st = time.time()
    #windows, clusters, clusteredWindows = getClusteredWindows(img,13,10)
    #en = time.time()
    #print("kesho ",(en-st)*1000)
    #st = time.time()
    #img = io.imread(image_name)
    #edged = cv2.Canny(img, 30, 200)
    #contours, hierarchy = cv2.findContours(edged,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    
    
    #img = np.ones(img.shape)*255
    #cv2.drawContours(img, contours, -1, (30, 255, 30), 1) 
   
    F5 = Chain_Codes_Counting_VEC(contours,True)
    #F1,F2,F3 = Chain_Codes_Counting_VEC(contours,False)
   
        
    
    F7 = np.array([])
    F8 = np.array([])
    F9 = np.array([])
    
    #for i in range(clusteredWindows.shape[0]):
     #   edged = cv2.Canny(clusteredWindows[i].astype(np.uint8), 30, 200)
      #  contours, hierarchy = cv2.findContours(edged,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
       # F7_local , F8_local , F9_local  = Chain_Codes_Counting_VEC(contours,False)
      
            
        #F7 = np.append(F7_local,F7)
        #F8 = np.append(F8_local,F8)
        #F9 = np.append(F9_local,F9)
    
    #en = time.time()
    #print((en-st)*1000)
    return F5
    
    return F1,F2,F3,F4,F5
    return F1,F2,F3,F4,F5,F7,F8,F9
    
    




   

    
    
    