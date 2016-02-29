import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# # get temperature from filename

# to get all filenames with extension .z in working directory
# filenames = os.listdir()

list_of_filenames = []
column_names = ['Freq', 'Ampl', 'Bias', 'Time(Sec)', 'Real(Z)', '-Im(Z)', 'GD', 'Err', 'Range']
data_frames = {}
for file in os.listdir():
    if file.endswith('.z'):
        filename = str.split(file, '.')[0]
        list_of_filenames.append(filename)
        df = pd.read_csv(file, skiprows=10, sep=',', names=column_names)
        data_frames[filename] = df.abs()
# df.rename(columns={0: temperature}, inplace=True)

# Remove unused columns
unused_columns = ['Ampl', 'Bias', 'Time(Sec)', 'GD', 'Err', 'Range']
for column in unused_columns:
    for df in data_frames:
        data_frames[df].drop(column, axis=1, inplace=True)

# normalize by sample dimension
diameter = float(input("sample diameter (cm)? "))
thickness = float(input("sample thickness (cm)? "))
geometric_factor = (np.pi * diameter ** 2 / 4) / thickness

for df in data_frames:
    data_frames[df]['Real(Z)'] /= geometric_factor  # normalizes impedance by geometric factor
    data_frames[df]['-Im(Z)'] /= geometric_factor
    data_frames[df]['log_freq'] = np.log10(data_frames[df]['Freq']).round(1)  # adds column to each df with log(f)

# Grab DataFrame rows where column has integer values
# and creates new dataframe
integers = range(-1, 7, 1)
log_freq_df = {}
for df in data_frames:
    log_freq_df[df] = data_frames[df][data_frames[df].log_freq.isin(integers)]

# # Plotting

# to create pannel use this instead
# fig = plt.figure()
# for index,df in enumerate(data_frames):
#     ax = fig.add_subplot(2,2,index+1)

for df in data_frames:
    temperature = df
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data_frames[df]['Real(Z)'], data_frames[df]['-Im(Z)'].abs(),
               color='blue',
               alpha=0.5,
               s=70)
    ax.scatter(log_freq_df[df]['Real(Z)'], log_freq_df[df]['-Im(Z)'].abs(),
               color='white',
               alpha=0.9,
               s=70)
    ax.set_aspect('equal')      # orthonormal axis
    plt.xlabel('Z\' (ohm)')
    plt.ylabel('-Z\'\' (ohm)')
    plt.title(temperature)
    plt.grid(True)
    plt.axis(xmin=0, ymin=0)

    # annotations
    x_ann = log_freq_df[df]['Real(Z)'].tolist()  # gets column 'Real('Z') into a list
    y_ann = log_freq_df[df]['-Im(Z)'].tolist()
    label = log_freq_df[df]['log_freq'].tolist()

    for i in range(len(x_ann)):
        plt.text(x_ann[i], y_ann[i], label[i])
plt.show()

