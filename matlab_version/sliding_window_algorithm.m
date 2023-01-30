%% sliding_window_algorithm.m
% This function implements sliding window algorithm by calculating the number
% of saccades of data within the window, sliding it throught the whole 
% dataset, and calculating the total number of detected saccades.

function [complete_detection_array, total_sacc] = sliding_window_algorithm(saccade_time_data, vel_denoise, vel_th, win_len)
% param saccade_time_data: the time data of the given dataset
% param vel_denoise: the denoised velocity signal
% param vel_th: the velocity threhsold set by the user
% param win_len: the length of the sliding window set by the user
% 
% return complete_detection_array: the array containing 0s and 1s. indicating 
% whether the saccade is detected or not.
% return total_sacc: the total number of saccades detected of the whole dataset.

% Create a label array for indexing
label = 1:length(vel_denoise);

% Create a dataframe containing time data, denoised velocity signal and labels
df = [saccade_time_data(:), vel_denoise(:), label(:)];

% Initialize total number of saccades, window start time
total_sacc = 0;
start_time = df(1, 1);

% Initialize a boolean variable for the loop
is_window = 1;

% Create complete detection array for the whole dataset
complete_detection_array = zeros(size(vel_denoise));

% Start detecting saccades with sliding window
while is_window
    % Set end time of the window
    end_time = start_time + win_len;

    % Obtain data in the window
    condition = saccade_time_data >= start_time & saccade_time_data < end_time;
    rolling_velocity = vel_denoise(condition);
    rolling_time = saccade_time_data(condition);
    rolling_label = label(condition);

    % Obtain detection array and saccade number of selected data
    [saccade_detection_array, saccade_detection, saccade_num] = threshold_in_window(rolling_velocity, rolling_time, rolling_label, vel_th);
    temp_label = [];
    for i = 1:length(saccade_detection_array)
        if saccade_detection_array(i) == 1
            temp_label(end+1) = saccade_detection(i,3);
        end
    end
    for j = 1:length(temp_label)
        index = temp_label(j);
        complete_detection_array(index) = 1;
    end

    % Calculate total saccade number
    total_sacc = total_sacc + saccade_num;
    start_time = end_time;

    % If window's end time exceeds the last time data, then the loop stops
    if start_time > max(df(end, 1))
        is_window = 0;
    end
end

end