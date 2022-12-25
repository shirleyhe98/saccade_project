import numpy as np
import scipy.signal as signal

def calc_derivative_signal(sacc_signal, Fs):
    """
    This function calculates the derivative of the input saccade signal to 
    calculate the velocities of each sample.

    @param signal: the input saccade signal
    @param Fs: the sampling rate (samples/second) of the input signal
    @return vel: the derivative of the input signal, i.e. the velocities
    """
    h = np.array([0.5, 0, -0.5])
    vel = signal.convolve(sacc_signal, Fs*h, mode = 'same')
    vel[0] = 0
    vel[-1] = 0
    return vel