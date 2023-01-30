from algorithms.sliding_window_algo import sliding_window_algorithm
from  algorithms.denoise_velocity import denoise_vel
from  algorithms.denoise_signal import denoise_signal
import scipy
import numpy as np

def CGTV_algo(Fs, original_signal, saccade_time, denoised_vel, num_iter=20):
    """
    This function implements the parametric model proposed by Dai et al which requires
    parameters alpha and beta be prescibed. Increasing alpha increases the sparsity of
    the estimated velocity time-series, whereas the value of beta increases the sparsity
    of the derivative of the acceleration. The parameter sigma is the estimated noise and
    D is the average saccade duration.
    The authors suggest that alpha and beta could be set as follow:
    if sampling frequency f <= 500, then:
        alpha = 0.016 f sigma
        beta = 0.008 f sprt(A) e^5D sigma
    else:
        alpha = (0.0032 f + 6.4) sigma
        beta = (0.0016 f + 3.2) sqrt(A) e^5D sigma
    
    Then implements the sliding window based velocity threshold algorithm.

    Reference: W. Dai, I. Selesnick, J.-R. Rizzo, J. Rucker and T. Hudson, 
    "Detection of normal and slow saccades using implicit piecewise polynomial 
    approximation," Journal of Vision, vol. 21, no. 6, pp. 1-18, 2021. 
    
    @param Fs: sampling rate of the singal.
    @param original_signal: original saccade position signal.
    @param saccade_time: time data of the given signal.
    @param denoised_vel: previously denoised velocity signal.
    @param num_iter: number of iteration times.

    @return denoised_signal: the position signal been denoised.
    @return detection_array: array containing 0s and 1s indicating whether a saccade is
    detected or not.
    @return total_sacs: number of total saccades being detected.
    """

    # Ask the user to enter preferred velocity threshold, window length and start time index
    vel_text = input("The default velocity threshold is 30 degrees/second. Do you want to change it (Y/n)? ")
    if vel_text == "Y":
        input_vel = input("Please enter your threshold: ")
        vel_th = int(input_vel)
    else:
        vel_th = 30
    
    window_text = input("The default window length is 10 seconds. Do you want to change it (Y/n)? ")
    if window_text == "Y":
        input_length = input("Please enter your window length: ")
        window_len = int(input_length)
    else:
        window_len = 10

    # Run sliding window VT algorithm to get detection array.
    detection_array, total_sacs = sliding_window_algorithm(saccade_time, denoised_vel, vel_th, window_len)

    # Get standard deviation for all fixations, this will be the sigma
    # Subtract average for each fixation
    fix_len = 0  # len of each fixation
    fixations = np.copy(original_signal)
    # detect_array = est_detect_array["complete detection"]
    for i in range(len(detection_array)):
        if detection_array[i] == 0:
            fix_len += 1
        elif fix_len != 0:
            fixations[i-fix_len:i] -= np.mean(original_signal[i-fix_len:i])
            fix_len = 0
    est_sigma = scipy.stats.tstd(fixations[detection_array == 0])

    # Get average duration
    est_total_dur = (detection_array == 1).sum()
    est_dur = est_total_dur / total_sacs / Fs

    # Get average amplitude
    est_v_smooth = denoise_vel(original_signal, Fs)
    est_total_amp = np.abs(est_v_smooth[detection_array == 1]).sum()
    est_amp = est_total_amp / total_sacs / Fs

    # Calculate the parameters alpha and beta.
    if Fs <= 500:
        alpha = 0.016 * Fs * est_sigma
        beta = 0.008 * Fs * np.sqrt(est_amp) * np.exp(5 * est_dur)
    else:
        alpha = (0.0032*Fs + 6.4) * est_sigma
        beta = (0.0016*Fs + 3.2) * np.sqrt(est_amp) * np.exp(5 * est_dur)
    
    # Denoise original saccade position signal
    denoised_signal = denoise_signal(original_signal, alpha, beta, num_iter)

    return denoised_signal, detection_array, total_sacs