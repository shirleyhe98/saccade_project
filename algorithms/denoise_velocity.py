import numpy as np
from algorithms.calc_derivative import calc_derivative_signal

def denoise_vel(position, Fs, movavg=0.02):
    """
    Call calc_derivative_signal function to calculate the derivative of position
    signal to obtain the velocities and smooth it out.

    @param position: the input saccade position signal
    @param Fs: the sampling rate (samples/second) of the input signal
    @param movavg: moving average in units of seconds, which is set to 0.02

    @return vel_denoise: the denoised velocity signal
    """

    # get velocity signal by taking the derivative of position signal
    velocity = calc_derivative_signal(position, Fs)

    # use moving average to smooth out the velocity
    movavg = int(movavg * Fs)
    vel_denoise = np.convolve(velocity, np.ones(movavg), 'same') / movavg

    return vel_denoise