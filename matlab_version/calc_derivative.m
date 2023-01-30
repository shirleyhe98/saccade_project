%% calc_derivative.m
% This function calculates the derivative of the input saccade signal to 
% calculate the velocities of each sample.

function vel = calc_derivative(sacc_signal, Fs)
% param sacc_signal: the input saccade signal
% param Fs: the sampling rate (samples/second) of the input signal
%
% return vel: the derivative of the input signal, i.e. the velocities

% Use central difference to calculate derivatives
central_diff = [1 0 -1] / 2;
vel = conv(sacc_signal, Fs*central_diff, "same");
vel(1) = NaN;
vel(end) = NaN;

end