import numpy as np
import cv2
import math

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

    return lane, m

def filter_lines(preprocessed_img):
    filtered_lines = []
    slopes = []
    count = 0
    lines = cv2.HoughLinesP(preprocessed_img, 2, np.pi / 180, 80, np.array([]), 50, 75)
    if lines is not None:
        for line in lines:
            try:
                for x1, y1, x2, y2 in line:
                    if np.sum(x2 - x1) != 0:
                        slope = np.sum(y2 - y1) / np.sum(x2 - x1)
                        if np.abs(slope) > 0.30 and np.abs(slope) < 0.80:
                            filtered_lines.append(line)
                            slopes.append(slope)
            except:
                pass
    count = len(filtered_lines)
    return filtered_lines, slopes, count
