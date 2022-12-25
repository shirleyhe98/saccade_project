import numpy as np
import pandas as pd
from algorithms.threshold_in_window import threshold_in_window

def sliding_window_algorithm(saccade_time_data, vel_denoise, vel_th, win_len):
    """
    This function implements sliding window algorithm by calculating the number of
    saccades of data within the window, sliding it throught the whole dataset, and
    calculating the total number of detected saccades.

    @param saccade_time_data: the time data of the given dataset
    @param vel_denoise: the denoised velocity signal
    @param vel_th: the velocity threhsold set by the user
    @param win_len: the length of the sliding window set by the user

    @return complete_detection_array: the array containing 0s and 1s. indicating 
    whether the saccade is detected or not.
    @return total_sacc: the total number of saccades detected of the whole dataset.
    """

    # Create a label array for indexing
    label = np.arange(0, len(vel_denoise)-1)

    # Create a dataframe containing time data, denoised velocity signal and labels
    df = pd.DataFrame.from_dict(zip(saccade_time_data, vel_denoise, label))
    df.columns = ['time', 'saccade velocity', 'label']

    # Initialize total number of saccades, window start time
    total_sacc = 0
    start_time = df["time"][0]

    # Initialize a boolean variable for the loop
    is_window = True

    # Create complete detection array for the whole dataset
    complete_detection_array = np.zeros_like(vel_denoise)

    # Start detecting saccades with sliding window
    while is_window:
        # Set end time of the window
        end_time = start_time + win_len

        # Obtain data in the window
        rolling_sample = df[(df["time"] >= start_time) & (df["time"] < end_time)]
        rolling_velocity = np.array(rolling_sample["saccade velocity"])
        rolling_time = np.array(rolling_sample["time"])
        rolling_label = np.array(rolling_sample["label"])

        # obtain detection array and saccade number of selected data
        saccade_detection_array, saccade_detection, saccade_num = threshold_in_window(rolling_velocity, rolling_time, rolling_label, vel_th)
        temp_label = []
        for i in range(len(saccade_detection_array)):
            if saccade_detection_array[i] == 1:
                temp_label.append(saccade_detection["label"][i])
        for l in temp_label:
            complete_detection_array[l] = 1
        
        # calculate total saccade number
        total_sacc += saccade_num
        start_time = end_time

        # if window's end time exceeds the last time data, then the loop stops
        if start_time > max(df["time"]):
            is_window = False
    
    complete_detection = pd.DataFrame.from_dict(zip(saccade_time_data, complete_detection_array, label))
    complete_detection.columns = ['time', 'complete detection', 'label']
    return complete_detection_array, total_sacc