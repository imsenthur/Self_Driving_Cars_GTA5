import numpy as np
import cv2
from PIL import ImageGrab
import pyautogui
import time
from Input import PressKey, ReleaseKey, W, A, S, D

def preprocess_image(orig_img):
    processed_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1 = 200, threshold2 = 300)
    return processed_img

for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(i)

PressKey(W)
print('Accelerating')
time.sleep(3)
ReleaseKey(W)
print('Decelerating')



while True:
    capture = np.array(ImageGrab.grab(bbox = (0,40,800,640)))
    preprocessed_image = preprocess_image(capture)
    cv2.imshow('Captured Frames',preprocessed_image)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
