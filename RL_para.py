import numpy as np
import scipy
from skimage import io, color
from skimage import exposure
import matplotlib.pyplot as plt
from skimage import data

import tensorflow as tf

img = data.astronaut()    # Load the image
img = color.rgb2gray(img)       # Convert the image to grayscale (1 channel)
tf.reset_default_graph()

ksize =3
max_thresh = 10
action_space = [-0.5, -0.45, -0.40, -0.35, -0.30, -0.25, -0.20, -0.15, -0.10, 0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.5]
Q = np.zeros([ksize * ksize, len(action_space)])
kernel = np.zeros([ksize,ksize])
learning_rate = 0.8
gamma = 0.75
n_episodes = 50
training_samples = 10
# you can use 'valid' instead of 'same', then it will not add zero padding


def reward_calc():
    reward = np.random.randint(0, 3)/ np.random.randint(1,3)
    return reward

def convolve_step(img, a, i, j, kernel):
    
    if j>2:
        i = i+1
        j = 0
        
    if i>2:
        i = 0
        
    kernel[i][j] = kernel[i][j] + action_space[a] * 0.5
    
    if np.abs(kernel[i][j]) >= max_thresh:
        kernel[i][j] = np.sign(kernel[i][j]) * max_thresh
        
    s = 3 * i + j
    j = j + 1
        
    return i, j, s

rlist = []
for i_episode in range(n_episodes):
    rALL = 0
    process_done = False
    i = 0
    j = 0
    s = 0
    for _ in range(training_samples):
        a = np.argmax(Q[s,:] + np.random.rand(1, len(action_space))*(1.0/(i_episode+1)))
        i, j, s_new = convolve_step(img, a, i, j, kernel, s)
        reward = reward_calc()
        #new_img, reward, process_done, s_new, _ = 
        Q[s, a] = Q[s, a] + learning_rate*(reward + gamma*np.max(Q[s_new, :]) - Q[s, a])
        rALL += reward
        #img = new_img
        s = s_new
        
        if process_done == True:
            break
    rlist.append(rALL)
    
    print(kernel)
    image_sharpen = scipy.signal.convolve2d(img, kernel, 'same')
    #print ('\n First 5 columns and rows of the image_sharpen matrix: \n', image_sharpen[:5,:5]*255)
    # Adjust the contrast of the filtered image by applying Histogram Equalization 
    image_sharpen_equalized = exposure.equalize_adapthist(image_sharpen/np.max(np.abs(image_sharpen)), clip_limit=0.03)
    plt.imshow(image_sharpen_equalized, cmap=plt.cm.gray)
    plt.axis('off')
    plt.show()
