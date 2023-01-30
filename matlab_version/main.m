%% main.m
% This program displays the original saccade data, denoised saccade data and 
% detected saccades.

clear
clc
%% Load antisaccade data

% Load antisaccade data
load("data/antisaccadeContPatient.mat");

patient_saccade = Dcell;
target_saccade = Tcell;

% Set patient group and first patient number
patient_group = 0;  % 0 for Control Group; 1 for Concussion Group
patient_num = 1;
Fs = 500;

if patient_group == 1
    % Concussion group
    patient_saccade_group = Dcell{1, 2};
    target_saccade_group = Tcell{1, 2};
    group = "Concussion Group";
else
    % Control group
    patient_saccade_group = Dcell{1, 1};
    target_saccade_group = Tcell{1, 1};
    group = "Control Group";
end

% Load actual data
patient_saccade_data = patient_saccade_group{patient_num, 1};
target_saccade_data = target_saccade_group{patient_num, 1};

horizontal_saccade = patient_saccade_data(:, 1);
horizontal_target_saccade = target_saccade_data(:, 1);
saccade_time_data = patient_saccade_data(:, 3);

%% Process data

% Denoise velocity
denoised_vel = denoise_vel(horizontal_saccade, Fs);

% Process signal with CGTV and sliding window VT
num_iter = 20;
[denoised_signal, complete_detection, total_sacs] = CGTV_algo(Fs, horizontal_saccade, saccade_time_data, denoised_vel, num_iter);

%% Plot data
figure(1)
hold on 
plot(saccade_time_data, complete_detection, 'g');
plot(saccade_time_data, horizontal_saccade, 'b');
plot(saccade_time_data, denoised_signal, 'r');
plot(saccade_time_data, horizontal_target_saccade, 'k');
legend('detection', 'original singal', 'denoised signal', 'target signal');
title(group + " patient number: " + patient_num + newline + total_sacs + " saccades are detected in total");