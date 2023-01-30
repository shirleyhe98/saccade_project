%% denoise_vel.m
% This function calls calc_derivative function to calculate the derivative of position
% signal to obtain the velocities and smooth it out.

function vel_denoise = denoise_vel(position, Fs)
% param position: the input saccade position signal
% param Fs: the sampling rate (samples/second) of the input signal
% param movavg: moving average in units of seconds, which is set to 0.02
% 
% return vel_denoise: the denoised velocity signal

% Get velocity signal by taking the derivative of position signal
velocity = calc_derivative(position, Fs);

% Use moving average to smooth out the velocity
moving_avg = 0.2;       % moving average in units of seconds, which is set to 0.02
mvavg = round(Fs * moving_avg);
mvavg_vec = ones([mvavg, 1]);
vel_denoise = conv(velocity, mvavg_vec, "same") ./ mvavg;
end