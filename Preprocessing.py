import numpy as np
import cv2
import math

def binary_mask(img, color_range):
    lower = np.array(color_range[0])
    upper = np.array(color_range[1])
    return cv2.inRange(img, color_range[0][0], color_range[1][0])

def binary_mask_apply(img, binary_mask):
    masked_image = np.zeros_like(img)
    for i in range(3):
        masked_image[:,:,i] = binary_mask.copy()
    return masked_image

def binary_mask_apply_color(img, binary_mask):
    return cv2.bitwise_and(img, img, mask = binary_mask)

def filter_by_color_ranges(img, color_ranges):
    result = np.zeros_like(img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    for color_range in color_ranges:
        color_bottom = color_range[0]
        color_top = color_range[1]

        if color_bottom[0] == color_bottom[1] == color_bottom[2] and color_top[0] == color_top[1] == color_top[2]:
            mask = binary_mask(gray_img, color_range)
        else:
            mask = binary_mask(hsv_img, color_range)

        masked_img = binary_mask_apply(img, mask)
        result = cv2.addWeighted(masked_img, 1.0, result, 1.0, 0.0)
        
    return result

def color_threshold(img, white_value):
    white = [[white_value, white_value, white_value], [255, 255, 255]]
    yellow = [[80,90,90], [120,255,255]]

    return filter_by_color_ranges(img, [white, yellow])

def equalize_histogram(img):
    img_yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def clache(img):
    c1 = cv2.createCLAHE(clipLimit = 3.0, tileGridSize = (8,8))
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(img_yuv)
    y_clache = c1.apply(y)
    img_yuv = cv2.merge((y_clache, u, v))
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def biliteral(img):
    return cv2.bilateralFilter(img, 13, 75, 75)

def unsharp_mask(image, blured):
    return cv2.addWeighted(blured, 1.5, blured, -0.5, 0, image)

def find_edges(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    v = np.median(gray_img)
    sigma = 0.33
    lower = int(max(180, (1.0 - sigma) * v))
    upper = int(min(350, (1.0 + sigma) * v))
    return canny(gray_img, lower, upper)

def preprocess_image(orig_img, white_value):
    processed_img = clache(orig_img)
    processed_img = color_threshold(processed_img, white_value)
    processed_img = find_edges(processed_img)
    
    processed_blurred_img = gaussian_blur(processed_img,3)
    processed_blurred_img = biliteral(processed_img)
    processed_img = unsharp_mask(processed_img, processed_blurred_img)
    count = cv2.countNonZero(processed_img)
    return processed_img, count

def hough_lines(image, white_value):
    if white_value < 150:
        return None

    processed_image = preprocess_image(image, white_value)
    houghed_lns = cv2.HoughLinesP(processed_image, 2, np.pi / 180, 100, np.array([]), 20, 100)
    
    if houghed_lns is None:
        return hough_lines(image, white_value - 5)
    
    return houghed_lns, processed_image

