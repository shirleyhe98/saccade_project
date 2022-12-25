import numpy as np
import pandas as pd
from algorithms.threshold_in_window_partial_data import threshold_in_window_partial_data

def sliding_window_algorithm_partial_data(saccade_time_data, vel_denoise, vel_th, win_len, start_index):
    """
    This function implements sliding window algorithm by calculating the number of
    saccades of data within the window. Given the time start index from user, it only shows
    the processed data from the start time index to the sum of it and window length.

    @param saccade_time_data: the time data of the given dataset
    @param vel_denoise: the denoised velocity signal
    @param vel_th: the velocity threhsold set by the user
    @param win_len: the length of the sliding window set by the user
    @param start_index: the start time of the sliding window set by the user

    @return complete_detection_array: the array containing 0s and 1s. indicating 
    whether the saccade is detected or not.
    @return total_sacc: the total number of saccades detected of the whole dataset.
    """

    # Create a label array for indexing
    label = np.arange(0, len(vel_denoise)-1)

    # Create a dataframe containing time data, denoised velocity signal and labels
    df = pd.DataFrame.from_dict(zip(saccade_time_data, vel_denoise, label))
    df.columns = ['time', 'saccade velocity', 'label']

    # Obtain signal start time
    start_time = df["time"][start_index]

    # Set end time of the window
    end_time = start_time + win_len

    # Obtain data in the window
    rolling_sample = df[(df["time"] >= start_time) & (df["time"] < end_time)]
    rolling_velocity = np.array(rolling_sample["saccade velocity"])

    # obtain detection array and saccade number of selected data
    saccade_detection_array, saccade_num = threshold_in_window_partial_data(rolling_velocity, vel_th)

    return saccade_detection_array, saccade_num