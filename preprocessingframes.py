import numpy as np
import cv2
from PIL import ImageGrab
import pyautogui
import time
from Input import PressKey, ReleaseKey, W, A, S, D
import ROI as roi
import COLORMASKING as mask



def draw_lines(img, lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0],coords[1]), (coords[2],coords[3]), [255,255,255], 3)
    except:
        pass

def colorcorrection(img):
    corrected_img = mask.equalize_histogram(img)
    corrected_img = mask.clache(corrected_img)
    corrected_img = mask.canny(corrected_img,200,300)
    corrected_img = mask.gaussian_blur(corrected_img,3)
    corrected_img = mask.bilateral(corrected_img)
    corrected_img = mask.unsharp_mask(corrected_img,100)
    return corrected_img   

def preprocess_image(orig_img):
    white_value =180
    processed_img = roi.crop(orig_img)
    processed_img = mask.color_threshold(processed_img, white_value)
    processed_img = colorcorrection(processed_img)
    
    #processed_img = cv2.Canny(orig_img, threshold1 = 200, threshold2 = 300)
    #processed_img = cv2.GaussianBlur(processed_img, (5,5), 0)

    #lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180, np.array([]), 100, 1)
    #draw_lines(processed_img, lines)
    return processed_img

while True:
    capture = np.array(ImageGrab.grab(bbox = (0,40,800,640)))
    preprocessed_image = preprocess_image(capture)
    cv2.imshow('Captured Frames',preprocessed_image)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
