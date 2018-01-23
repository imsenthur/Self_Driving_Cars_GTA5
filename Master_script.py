import numpy as np
import cv2
from PIL import ImageGrab
import pyautogui
import time
from Input import PressKey, ReleaseKey, W, A, S, D, go_straight, go_right, go_left, slow_down
import Preprocessing
import LaneFinder
import ScreenGrab
import CropFunctions

def self_drive():
    if left_lane_slope < 0 and right_lane_slope < 0:
        go_right()
        print('taking a right turn')
    elif left_lane_slope > 0 and right_lane_slope > 0:
        go_left()
        print('taking a left turn')
    else:
        go_straight()
        print('going straight')

filtered_lines = []
slopes = []
randomness = 500
i_rand = 0
s = 0
gamma = 0.35
learning_rate = 0.8
action_space = [160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220, 225, 230, 235, 240]
Q = np.zeros([len(action_space), len(action_space)])

def reward_calculator(count, line_count):
    reward = 0
    #8250 optimum,
    temp = (count - 8250)/10000
    if np.sign(temp) == 1:
        if np.abs(temp) > 0.5:
            reward = -5.0
        elif np.abs(temp) > 0.2:
            reward = -2.5
        else:
            reward = 3.0
    elif np.sign(temp) == -1:
        if np.abs(temp) > 0.2:
            reward = -2.0
        else:
            reward = 1.0
    if line_count < 2:
        reward += -5.0
    elif line_count < 10:
        reward += 3.0
    elif line_count > 15:
        reward += -5.0
    else:
        reward += 1.0
    #reward = np.sign(temp)
    return reward

def step(a, s, count, line_count):
    s = a
    reward = reward_calculator(count, line_count)
    return s, reward
    
while True:
    left_lane_slope = 0
    right_lane_slope = 0

    #Capturing and preprocessing
    capture = np.array(ScreenGrab.grab_screen(region=(0,40,800,640)))
    cropped_img = CropFunctions.crop(capture)
    
    #Rl begins
    a = np.argmax(Q[s,:] + np.random.rand(1, len(action_space))*(1.0/(i_rand+1)))
    preprocessed_img, count = Preprocessing.preprocess_image(cropped_img, action_space[a])
    filtered_lines, slopes, line_count = LaneFinder.filter_lines(preprocessed_img)
    s_new, reward = step(a, s, count, line_count)
    Q[s, a] = Q[s, a] + learning_rate * (reward + gamma*np.max(Q[s_new,:]) - Q[s,a])
    s = s_new
    if i_rand > randomness:
        i_rand = 0
                                                      
    LaneFinder.draw_lines(preprocessed_img, filtered_lines)
    cv2.imshow('houghlines',preprocessed_img)
    
    #cv2.imshow('Captured',preprocessed_img)
    #lines = Preprocessing.hough_lines(preprocessed_img)
    '''
    #finding lanes
    left_lines, right_lines, left_slopes, right_slopes = LaneFinder.left_right_lines(lines)
    left_lane, left_lane_slope = LaneFinder.find_the_lane(left_lines, left_slopes)
    right_lane, right_lane_slope = LaneFinder.find_the_lane(right_lines, right_slopes)

    print(left_lane_slope, right_lane_slope)

    #self-drive


    #output window
    print(left_lane_slope, right_lane_slope)
    LaneFinder.draw_lines(capture, left_lane)
    LaneFinder.draw_lines(capture, right_lane)
    cv2.imshow('Lanes on road',cv2.cvtColor(capture, cv2.COLOR_BGR2RGB))
    '''
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
