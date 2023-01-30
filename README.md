# Saccade Detection Project

This project is created for ECE-GY 9953 project course with Prof. Ivan Selesnick.


The goal of this project is to modify an algorithm proposed by [1]. The algorithm is created to denoise the time series saccade data and implement a simple velocity threshold algorithm to detect normal and slow sacades.

The modified algorithm keeps the denoising part but implements a sliding window with a user-prompt velocity threshold, a user-prompt window length and a user-prompt window start timestamp to detect saccades on antisaccade data. 

## Python
The algorithm functions are divided into pieces, i.e. each function is placed in a separater file. To run the program, please run two GUI programs. saccade_GUI.py displays patients' antisaccade data and enables the user to adjust window length and velocity threshold used in the algorithm. saccade_GUI_partial.py displays part of the saccade data that could enable doctors to see the saccades closely. It enables the user to adjust the timestamp that window starts, the length of window and velocity threshold used in the algorithm.

## Matlab
The Matlab code programs are stored in the folder "matlab_version". Simply run the file main.m will run all the programs. The file produces a plot showing original signal, denoised signal, target signal, and detections, which is similar to the Python plot. To switch between patient groups or patients, simply modify the variables patient_group or patient_num in the main.m file.

The source code is based on [1] and [2].

[1] W. Dai, I. Selesnick, J.-R. Rizzo, J. Rucker and T. Hudson, "Detection of normal and slow saccades using implicit piecewise polynomial approximation," Journal of Vision, vol. 21, no. 6, pp. 1-18, 2021. 

[2] J. Wu and M. Baig, "DP2 Project: Application of CGTV Algorithm on Processing Saccade Data," 2022.
