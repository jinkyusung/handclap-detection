import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

import detect
import misc


# Parameters
FILE_PATH = './videos/clap_cropped.mp4'
TD = 0.15
TA = 125
TC = 125
TR = 0
ANSWER = [(65, 75), (185, 195), (230, 240)]


def main():
    # Mode Selection
    if len(sys.argv) > 1 and sys.argv[1] == '-r':  
        # Real-time Mode: With WebCam
        cap = cv2.VideoCapture(0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        distances, included_angles, left_fingers_mean, right_fingers_mean, \
        left_x_directs, left_y_directs, left_z_directs, right_x_directs, right_y_directs, right_z_directs = detect.pattern(cap, 0.6, 0.6)

    elif len(sys.argv) > 1 and sys.argv[1] == '-l':
        # Load Mode: existing detection data 
        cap = cv2.VideoCapture(FILE_PATH)
        fps = cap.get(cv2.CAP_PROP_FPS)
        distances = np.load('./data/distances.npy')
        included_angles = np.load('./data/included_angles.npy')
        left_fingers_mean = np.load('./data/left_fingers_mean.npy')
        right_fingers_mean = np.load('./data/right_fingers_mean.npy')
        left_x_directs = np.load('./data/left_x_directs.npy')
        left_y_directs = np.load('./data/left_y_directs.npy')
        left_z_directs = np.load('./data/left_z_directs.npy')
        right_x_directs = np.load('./data/right_x_directs.npy')
        right_y_directs = np.load('./data/right_y_directs.npy')
        right_z_directs = np.load('./data/right_z_directs.npy')

    elif len(sys.argv) > 1 and sys.argv[1] == '-s':
        # Save Mode: Detects and extracts data from video, and save them as .npy files
        cap = cv2.VideoCapture(FILE_PATH)
        fps = cap.get(cv2.CAP_PROP_FPS)
        distances, included_angles, left_fingers_mean, right_fingers_mean, \
        left_x_directs, left_y_directs, left_z_directs, right_x_directs, right_y_directs, right_z_directs = detect.pattern(cap, 0.6, 0.8)
        np.save('./data/distances.npy', distances)
        np.save('./data/included_angles.npy', included_angles)
        np.save('./data/left_fingers_mean.npy', left_fingers_mean)
        np.save('./data/right_fingers_mean.npy', right_fingers_mean)
        np.save('./data/left_x_directs.npy', left_x_directs)
        np.save('./data/left_y_directs.npy', left_y_directs)
        np.save('./data/left_z_directs.npy', left_z_directs)
        np.save('./data/right_x_directs.npy', right_x_directs)
        np.save('./data/right_y_directs.npy', right_y_directs)
        np.save('./data/right_z_directs.npy', right_z_directs)
    
    else:
        # Basic Mode: Only Detects and extracts data from video
        cap = cv2.VideoCapture(FILE_PATH)
        fps = cap.get(cv2.CAP_PROP_FPS)
        distances, included_angles, left_fingers_mean, right_fingers_mean, \
        left_x_directs, left_y_directs, left_z_directs, right_x_directs, right_y_directs, right_z_directs = detect.pattern(cap, 0.6, 0.8)
    # Plot
    outliers_distances = detect.outliers(
        distances, included_angles, left_fingers_mean, right_fingers_mean, \
        left_x_directs, left_y_directs, left_z_directs, right_x_directs, right_y_directs, right_z_directs, \
        t_dist=TD, t_angle=TA, t_curv=TC, t_dir=TD
    )
    misc.highlight_outliers('distances', distances, outliers_distances, fps, ANSWER)
    misc.highlight_outliers('Inter', included_angles, None, fps, None)
    misc.highlight_outliers('left-mean', left_fingers_mean, None, fps, None)
    misc.highlight_outliers('right-mean', right_fingers_mean, None, fps, None)
    misc.highlight_outliers('left-x', left_x_directs, None, fps, None)
    misc.highlight_outliers('left-y', left_y_directs, None, fps, None)
    misc.highlight_outliers('left-z', left_z_directs, None, fps, None)
    misc.highlight_outliers('right-x', right_x_directs, None, fps, None)
    misc.highlight_outliers('right-y', right_y_directs, None, fps, None)
    misc.highlight_outliers('right-z', right_z_directs, None, fps, None)
    plt.show()


if __name__ == '__main__':
    main()
    