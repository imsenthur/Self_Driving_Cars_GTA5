import numpy as np
import scipy.signal
import cv2
import time
import ScreenGrab
import pyautogui

while True:
    img = np.array(ScreenGrab.grab_screen(region=(0,40,800,640)))
    cv2.imshow('captured_img',cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ### apply sharpen filter to the original image
    ##sharpen_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    ##image_sharpen = scipy.signal.convolve2d(img, sharpen_kernel, 'valid')
    ### apply edge detection filter to the sharpen image
    ##edge_kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    ##edges = scipy.signal.convolve2d(image_sharpen, edge_kernel, 'valid')
    ### apply blur filter to the edge detection filtered image
    ##blur_kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])/9.0;
    ##denoised = scipy.signal.convolve2d(edges, blur_kernel, 'valid')
    ### Adjust the contrast of the filtered image by applying Histogram Equalization
    ##denoised_equalized = exposure.equalize_adapthist(denoised/np.max(np.abs(denoised)), clip_limit=0.03)
    ##plt.imshow(denoised_equalized, cmap=plt.cm.gray)    # plot the denoised_clipped
    ##plt.axis('off')
    ##plt.show()
