import numpy as np
import cv2
from PIL import ImageGrab
import pyautogui
import time
from Input import PressKey, ReleaseKey, W, A, S, D

def draw_lines(img, lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0],coords[1]), (coords[2],coords[3]), [255,255,255], 3)
    except:
        pass

def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def preprocess_image(orig_img):
    processed_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1 = 200, threshold2 = 300)
    processed_img = cv2.GaussianBlur(processed_img, (5,5), 0)
    
    vertices = np.array([[10,550], [10,325], [300,325], [500,325], [800,325], [800,550]])
    processed_img = roi(processed_img, [vertices])

    lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180, np.array([]), 100, 5)
    draw_lines(processed_img, lines)
    return processed_img

while True:
    capture = np.array(ImageGrab.grab(bbox = (0,40,800,640)))
    preprocessed_image = preprocess_image(capture)
    cv2.imshow('Captured Frames',preprocessed_image)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
