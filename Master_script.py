import numpy as np
import cv2
from PIL import ImageGrab
import pyautogui
import time
from Input import PressKey, ReleaseKey, W, A, S, D
import Preprocessing

def draw_lines(img, lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0],coords[1]), (coords[2],coords[3]), [180, 253, 11], 3)
    except Exception as e:
        pass

def left_right_lines(lines):
    lines_all_left = []
    lines_all_right = []
    slopes_left = []
    slopes_right = []

    for line in lines:
        try:
            for x1, y1, x2, y2 in line:
                if np.sum(x2 - x1) != 0:
                    slope = np.sum(y2 - y1) / np.sum(x2 - x1)
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

    return filtered_left_lns, filtered_right_lns, slopes_left, slopes_right

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

def find_the_lane(lines, slopes):
    m = np.mean(slopes)
    lane = [[]]
    last_diff = 0
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = np.sum(y2 - y1) / np.sum(x2 - x1)
            diff = abs(m-slope)
            if last_diff == 0:
                lane[0] = line
            elif diff < last_diff:
                lane[0] = line
            last_diff = diff

    return lane

while True:
    white_value = 230
    capture = np.array(ImageGrab.grab(bbox = (0,40,800,640)))
    lines, processed_img = Preprocessing.hough_lines(capture, white_value)
    left_lines, right_lines, left_slopes, right_slopes = left_right_lines(lines)
    
    left_lane = find_the_lane(left_lines, left_slopes)
    right_lane = find_the_lane(right_lines, right_slopes)
    draw_lines(capture, left_lane)
    draw_lines(capture, right_lane)
    cv2.imshow('Lanes on road',cv2.cvtColor(capture, cv2.COLOR_BGR2RGB))
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
