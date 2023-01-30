%% threshold_in_window.m
% This function calculate the number of saccades in a selected set of velocity
% data within the window.

function [detection_array, saccade_detection, saccade_num] = threshold_in_window(vel_denoise, time_data, label_data, vel_th)
% param vel_denoise: the denoised and smoothed velocity signal
% param time_data: the part of array of time corresponding to velocity signal 
% obtained from the given data
% param label_data: the part of array of labels corresponding to velocity 
% signal created for all signal data
% param vel_th: the velocity threshold that used to determine saccades and fixations
% 
% return detection_array: the array of 0s and 1s indicating fixations 
% and saccades respectively within the window
% return saccade_detection: the multidimensional array containing time, detection array 
% and labels within the window
% return saccade_num: the number of detected saccades within the window

% Create empty detection array and saccade number within the window
detection_array = zeros(size(vel_denoise));
saccade_num = 0;

% Obtain peak velocities by using islocalmax function
peak_vel_found = islocalmax(vel_denoise, 'MinSeparation', 25);
peak_vel = vel_denoise(peak_vel_found);

% peak_vel_found = diff(sign(diff(vel_denoise)))== -2;
% peak_vel = vel_denoise(peak_vel_found);

% Start saccade detection within the window
for i = 1:length(peak_vel)
    vel = peak_vel(i);
    % if velocity is greater than or equal to the threshold, then saccade detects
    if abs(vel) >= vel_th
        index = vel_denoise==vel;
        detection_array(index) = 1;
        saccade_num = saccade_num + 1;
    end
end

saccade_detection = [time_data(:), detection_array(:), label_data(:)];

end