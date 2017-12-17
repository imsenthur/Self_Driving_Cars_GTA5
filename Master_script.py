import numpy as np
import cv2
from scipy.stats import linregress
from PIL import ImageGrab
import pyautogui
import time
from Input import PressKey, ReleaseKey, W, A, S, D
import Preprocessing

def draw_lines(img, lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0],coords[1]), (coords[2],coords[3]), [255,255,255], 3)
    except:
        pass

def left_right_lines(lines):
    lines_all_left = []
    lines_all_right = []
    slopes_left = []
    slopes_right = []

    for line in lines:
        try:
            for x1, y1, x2, y2 in line:
                if x2 - x1 != 0:
                    slope = (y2 - y1) / (x2 - x1)
                if slope > 0:
                    lines_all_right.append(line)
                    slopes_right.append(slope)
                elif slope < 0:
                    lines_all_left.append(line)
                    slopes_left.append(slope)
        except:
            pass

    filtered_left_lns = filter_lines_outliers(lines_all_left, slopes_left, True)
    filtered_right_lns = filter_lines_outliers(lines_all_right, slopes_right, False)

    return filtered_left_lns, filtered_right_lns

def filter_lines_outliers(lines, slopes, is_left, min_slope = 0.1, max_slope = 0.9):
    if len(lines) < 2:
        return lines

    lines_no_outliers = []
    slopes_no_outliers = []

    for i, line in enumerate(lines):
        slope = slopes[i]
        if min_slope < abs(slope) < max_slope:
            lines_no_outliers.append(line)
            slopes_no_outliers.append(slope)

    slope_median = np.median(slopes_no_outliers)
    slope_std_deviation = np.std(slopes_no_outliers)
    filtered_lines = []

    for i, line in enumerate(lines_no_outliers):
        slope = slopes_no_outliers[i]
        intercepts = np.median(line)

        if slope_median - 2 * slope_std_deviation < slope < slope_median + 2 * slope_std_deviation:
            filtered_lines.append(line)

    return filtered_lines

def median(lines):   
    xs = []
    ys = []
    xs_med = []
    ys_med = []
    m = 0
    b = 0

    for line in lines:
        for x1, y1, x2, y2 in line:
            xs += [x1, x2]
            ys += [y1, y2]
            
    m, b, r_value_left, p_value_left, std_err = linregress(xs, ys)

    print(m, b)
    for line in lines:
        for x1, y1, x2, y2 in line:
            y1 = m*x1 + b
            y2 = m*x2 + b
    
    return lines

while True:
    white_value = 230
    capture = np.array(ImageGrab.grab(bbox = (0,40,800,640)))
    lines, processed_img = Preprocessing.hough_lines(capture, white_value)
    left_lines, right_lines = left_right_lines(lines)
##    draw_lines(processed_img, right_lines)
##    draw_lines(processed_img, left_lines)
    left_lane = median(left_lines)
    draw_lines(processed_img, left_lane)
    cv2.imshow('Processed Frame',processed_img)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
