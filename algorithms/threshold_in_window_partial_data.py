import numpy as np
import scipy.signal as signal

def threshold_in_window_partial_data(vel_denoise, vel_th):
    """
    This function calculate the number of saccades in a selected set of velocity
    data within the window.

    @oaram vel_denoise: the denoised and smoothed velocity signal
    @param time_data: the part of array of time corresponding to velocity signal obtained 
    from the given data
    @param label_data: the part of array of labels corresponding to velocity signal created
    for all signal data
    @param vel_th: the velocity threshold that used to determine saccades and fixations

    @return detection_array: the array of 0s and 1s indicating fixations 
    and saccades respectively within the window
    @return saccade_num: the number of detected saccades within the window
    """

    # create empty detection array and saccade number within the window
    detection_array = np.zeros_like(vel_denoise)
    saccade_num = 0

    # Obtain peak velocities by using argrelextrema function
    peak_vel = vel_denoise[signal.argrelextrema(vel_denoise, np.greater)]

    # convert array to list
    vel_denoise = list(vel_denoise)

    # Start saccade detection within the window
    for vel in peak_vel:
        # if velocity is greater than or equal to the threshold, then saccade detects
        if abs(vel) >= vel_th:
            # obtain the index of the velocity
            index = vel_denoise.index(vel)
            detection_array[index] = 1
            saccade_num += 1
    
    return detection_array, saccade_num