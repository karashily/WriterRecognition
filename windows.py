import cv2
import matplotlib.pyplot as plt
import skimage
import itertools
import numpy as np

def getWindows(img, windowL):
    _, thr = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    outputWindows = []
    imgWindows = skimage.util.shape.view_as_windows(img, (windowL, windowL))
    thrWindows = skimage.util.shape.view_as_windows(thr, (windowL, windowL))
    for i in range(thr.shape[0]//windowL):
        for j in range(thr.shape[1]//windowL):
            for ii, jj in itertools.product(range(windowL), range(windowL)): 
                    if i*windowL+ii < imgWindows.shape[0] and j*windowL+jj < imgWindows.shape[1] and \
                                thrWindows[i*windowL+ii, j*windowL+jj, 0, 0]==0:
                        outputWindows.append(imgWindows[i*windowL+ii, j*windowL+jj])
                        break
    outputWindows = np.array(outputWindows)
    return outputWindows

def plotImages(images):
    '''plot the images in the batch'''
    fig = plt.figure(figsize=(25, 5))
    # display 20 images
    for idx in np.arange(len(images)):
        ax = fig.add_subplot(2, len(images)/2, idx+1, xticks=[], yticks=[])
        plt.imshow(images[idx], 'gray')
    plt.show()