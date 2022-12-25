# saccade_GUI.py
# This program constructs a GUI to display original saccade data, denoised saccade data and 
# detected saccades.

import scipy.io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')

from algorithms.denoise_velocity import denoise_vel
from algorithms.CGTV_algo import CGTV_algo
import numpy as np
import pandas as pd

def update_displayed_data():
    """
    This function updates the group of patients and the number of patients displayed on
    the GUI.
    """
    if patient_group == 1:
        patient_label.configure(text="%s group: patient number: %d" % ('Concussion', patient_num))
    else:
        patient_label.configure(text="%s group: patient number: %d" % ('Control', patient_num))

def load_antisaccade_data(patient_group, patient_num, Fs):
    """
    This function loads the antisaccade data and find the saccade detection array and total
    number of detected saccades, preparing for the plot.

    @param patient_group: group that the patient belongs to (Control/Concussion)
    @param patient_num: number of the patient
    @param Fs: sampling rate of the signal
    """

    global total_sacs, data_plt_map
    patient_saccade = data['Dcell']
    target_saccade = data['Tcell']

    if patient_group:
        # Concussion group
        patient_saccade_group = patient_saccade[0, 1]
        target_saccade_group = target_saccade[0, 1]
    else:
        # Control group
        patient_saccade_group = patient_saccade[0, 0]
        target_saccade_group = target_saccade[0, 0]
    
    # Load actual data
    patient_saccade_data = patient_saccade_group[patient_num-1, 0]
    target_saccade_data = target_saccade_group[patient_num-1, 0]

    horizontal_saccade = patient_saccade_data[:, 0]
    horizontal_target_saccade = target_saccade_data[:, 0]
    saccade_time_data = patient_saccade_data[:, 2]

    # Denoise velocity
    denoised_vel = denoise_vel(horizontal_saccade, Fs)

    # Create a dataframe of target saccade positions
    label = np.arange(0, len(denoised_vel)-1)
    df = pd.DataFrame.from_dict(zip(saccade_time_data, horizontal_target_saccade, label))
    df.columns = ['time', 'target position', 'label']

    # Process signal with CGTV and sliding window VT
    denoised_signal, complete_detection, total_sacs = CGTV_algo(Fs, horizontal_saccade, saccade_time_data, denoised_vel)
    # Data that needed to be displayed
    data_plt_map = {}
    data_plt_map['detection'] = {'visible_var': detection_line, 'params': [saccade_time_data, complete_detection, 'g']}
    data_plt_map['original singal'] = {'visible_var': original_line, 'params': [saccade_time_data, horizontal_saccade, 'b']}
    data_plt_map['denoised signal'] = {'visible_var': denoised_line, 'params': [saccade_time_data, denoised_signal, 'r']}
    data_plt_map['target signal'] = {'visible_var': target_line, 'params': [saccade_time_data, horizontal_target_saccade, 'orange']}

def set_GUI_plot(figure):
    """
    This function sets the GUI plot to be displayed.

    @param figure: matplotlib figure to be displayed
    """
    # plot setting
    axis0 = figure.add_subplot(1, 1, 1)
    axis0.set_ylabel("Position (deg)")
    axis0.set_xlabel("Time (s)")
    axis0.set_title("CGTV: %d saccades detected in total" % total_sacs, fontsize=16)

    # plot detection, original signal, target signal, and denoised signal
    for key in data_plt_map:
        if data_plt_map[key]['visible_var'].get():
            axis0.plot(*data_plt_map[key]['params'], label=key)
    
    # show legend
    axis0.legend()

def update_GUI_plot():
    """
    This function updates the displayed plot.
    """
    axis0, = figure.axes
    axis0.remove()
    set_GUI_plot(figure)
    figure.canvas.draw()

def update_patient():
    """
    This function updates the patient data as the patient number changes.
    """
    # update dropdown menu to select patient
    # patient_option_var: tkinter string variable
    patient_option_var.set(str(patient_num))
    
    # update patient display info
    update_displayed_data()

    # load new patient data
    load_antisaccade_data(patient_group, patient_num, Fs)

    # plot new data
    update_GUI_plot()

def change_patient(patient_num_str):
    """
    This function changes the patient data when patient number is changed.

    @param patient_num_str: patient number in string
    """
    global patient_num
    # change patient when patient number is changed
    if patient_num != int(patient_num_str):
        patient_num = int(patient_num_str)
        update_patient()

def change_patient_group():
    """
    This function updates the group that patient belongs to when patient group is changed.
    """
    # If button is pressed, then patient group is control; if button is pressed again, the patient group should
    # change to concussion group

    global patient_group
    if group_button['relief'] == 'sunken':
        group_button['relief'] = 'raised'
        patient_group = 0
        button_text.set("Control")
        update_patient()
    else:
        group_button['relief'] = 'sunken'
        patient_group = 1
        button_text.set("Concussion")
        update_patient()

def left_control_button():
    """
    This function sets how th left button is used to switch to previous patients.
    """
    global patient_num
    # Switch to previous patient when left button is pressed
    patient_num -= 1
    if patient_num < PATIENT_NUM_MIN:
        patient_num = PATIENT_NUM_MIN
    else:
        update_patient()

def right_control_button():
    """
    This function sets how th left button is used to switch to later patients.
    """
    global patient_num
    # Switch to next patient when right button is pressed
    patient_num += 1
    if patient_num > PATIENT_NUM_MAX:
        patient_num = PATIENT_NUM_MAX
    else:
        update_patient()

PATIENT_NUM_MIN = 1
PATIENT_NUM_MAX = 21

# Load antisaccade data
data = scipy.io.loadmat("data/antisaccadeContPatient.mat")

# set patient group and first patient number
patient_group = 0
patient_num = 1
Fs = 500

root = tk.Tk()
root.title('A Parametric Saccade Model with Sliding Window Velocity Threshold')

# Bind left and right button to switch patients
root.bind("<Left>", left_control_button)
root.bind("<Right>", right_control_button)

# Create frames
top_frame = tk.Frame(root)
bottom_frame = tk.Frame(root)
tools_frame = tk.Frame(bottom_frame)

# create frame widgets
patient_label = tk.Label(top_frame, text="", font=('Arial', 20))
update_displayed_data()

button_text = tk.StringVar()
group_button = tk.Button(top_frame, textvariable=button_text, command=change_patient_group)
button_text.set("Control")

patient_option_var = tk.StringVar(root, '1')
patient_option = tk.OptionMenu(top_frame, patient_option_var,
                               *[str(i+1) for i in range(PATIENT_NUM_MAX)],
                               command=change_patient)

# tk integer to control visibility of the plot
detection_line = tk.IntVar(value=1)
original_line = tk.IntVar(value=1)
denoised_line = tk.IntVar(value=1)
target_line = tk.IntVar(value=1)
load_antisaccade_data(patient_group, patient_num, Fs)

# create plot 
figure = matplotlib.figure.Figure(figsize=(9, 4))
figure.subplots_adjust(wspace=0.3)
figure.subplots_adjust(left=0.07, right=0.95, bottom=0.15)
set_GUI_plot(figure)

# Canvas widget for matplotlib and tkinter
canvas = FigureCanvasTkAgg(figure, master=root)
canvas_widget = canvas.get_tk_widget()

# Left and right buttons to switch patients
left_key = tk.Button(tools_frame, text='<<', command=left_control_button)
right_key = tk.Button(tools_frame, text='>>', command=right_control_button)

# checkbuttons to show/hide data groups
button_disp = []
for key in data_plt_map:
    button = tk.Checkbutton(tools_frame, text=key,
                         variable=data_plt_map[key]['visible_var'],
                         onvalue=1,
                         offvalue=0,
                         command=update_GUI_plot)
    button_disp.append(button)

# Place frames
top_frame.pack()
canvas_widget.pack()
bottom_frame.pack(side='bottom')
tools_frame.pack()

# Place widgets
patient_label.pack(side=tk.LEFT)
group_button.pack(side=tk.LEFT)
patient_option.pack(side=tk.LEFT)

left_key.pack(side=tk.LEFT)
for button in button_disp:
    button.pack(side=tk.LEFT)
right_key.pack(side=tk.LEFT)

root.mainloop()