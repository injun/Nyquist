import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# # get temperature from filename
# to get all filenames with extension .z in working directory
# filenames = os.listdir()
list_of_files = os.listdir()
filename = []
for file in list_of_files:
    if file.endswith(".z"):
        filename.append(str.split(file, '.')[0])    # splits into a list at the '.', gets data at index 0
print(filename)


# # import impedance files into dataframe
file_names = []
data_frames = {}
column_names = ['Freq', 'Ampl', 'Bias', 'Time(Sec)', 'Real(Z)', '-Im(Z)', 'GD', 'Err', 'Range']
for filename in os.listdir():
    if filename.endswith('.z'):
        temperature = str.split(filename, '.')[0]
        file_names.append(temperature)
        df = pd.read_csv(filename, skiprows=10, sep=',', names=column_names)
        # df.rename(columns={0: temperature}, inplace=True)
        data_frames[temperature] = df.abs()   # modulus of Im(Z)

data_frames['292'].head()


# # Remove unused columns
# -- there's a way to import specific columns, but can't get it to work
unused_columns = ['Ampl', 'Bias', 'Time(Sec)', 'GD', 'Err', 'Range']
for column in unused_columns:
    for df in data_frames:
        data_frames[df].drop(column, axis=1, inplace=True)

data_frames['293'].head()        

# # Alternative method to import csv into dataframe 
# column_names = ['Freq', 'Ampl', 'Bias', 'Time(Sec)', 'Real(Z)', '-Im(Z)', 'GD', 'Err', 'Range']
# frame = pd.DataFrame()
# list_= []
# for file in list_of_files:
#    if file.endswith('.z'):
#         df = pd.read_csv(file, skiprows=10, sep=',', names=column_names)
#         list_.append(df)
        
#     unused_columns = ['Ampl', 'Bias', 'Time(Sec)', 'GD', 'Err', 'Range']
#     for column in unused_columns:
#         impedance.drop(column, axis=1, inplace=True)
# there's a way to import specific columns, but can't get it to work
# deleted unused colums
# unused_columns = ['Ampl', 'Bias', 'Time(Sec)', 'GD', 'Err', 'Range']
# for column in unused_columns:
#     impedance.drop(column, axis=1, inplace=True)

# impedance = impedance.abs()    # signal of imaginary impedance

# normalize by sample dimension

diameter = float(input("sample diameter (cm)? "))
thickness = float(input("sample thickness (cm)? "))
geometric_factor = (np.pi*diameter**2/4)/thickness

for df in data_frames:
    data_frames[df]['Real(Z)'] /= geometric_factor
    data_frames[df]['-Im(Z)'] /= geometric_factor

data_frames['292'].head()


# # calculate log frequency for plot labels

# this will add a new column to each dataframe
# avoids errors if measuring frequency range changes within data set

for df in data_frames:
    data_frames[df]['log_freq'] = np.log10(data_frames[df]['Freq']).round(1)

data_frames['292'].head()


# In[215]:

print(file_names)


# In[225]:

# TODO:
# plot a second series in the same graph, where the points correspond to log of frequencies are close to an integer 
# to the second decimal case
# label those points with the integer log value

# Grab DataFrame rows where column has certain values
integers = range(-1,7,1)
log_freq_df = {}

for i in file_names:
    for df in data_frames:
        log_freq_df[i] = data_frames[df][data_frames[df].log_freq.isin(integers)].astype(int)


# In[227]:

print(data_frames['292']['log_freq'])
print(log_freq_df['292'].head())


# In[218]:

plt.scatter(data_frames['292']['Real(Z)'], data_frames['292']['-Im(Z)'])
plt.scatter(log_freq_df['292']['Real(Z)'], log_freq_df['292']['-Im(Z)'])


# In[204]:

plt.scatter(data_frames['292']['Real(Z)'], data_frames['292']['-Im(Z)'], 
            color='blue',
            alpha=0.5,
            s=60,
            label=temperature + ' ºC')
plt.scatter(log_freq_df['292']['Real(Z)'], log_freq_df['292']['-Im(Z)'],
            color="white",
            alpha=1,
            label='_nolegend_', 
            s=60)
plt.xlabel('Z\' (ohm)')
plt.ylabel('-Z\'\' (ohm)')
plt.title(temperature)
plt.grid(True)
plt.axis(xmin=0, ymin=0) 
# plt.legend(loc=(0,0.9))

# print log_freq as labels
# TODO: change to annotations + arrow function

x_ann = log_freq_df['292']['Real(Z)'].tolist()    # gets column 'Real('Z') into a list
y_ann = log_freq_df['292']['-Im(Z)'].tolist()
label = log_freq_df['292']['log_freq'].tolist()

for i in range(len(x_ann)):
    plt.text(x_ann[i], y_ann[i], label[i])


# In[44]:

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot([1, 2, 3], [1, 2, 3]);
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot([1, 2, 3], [3, 2, 1]);
plt.show()


# In[47]:



fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(impedance['Real(Z)'], impedance['-Im(Z)'].abs())
plt.xlabel('Z\' (ohm)')
plt.ylabel('-Z\'\' (ohm)')
plt.title(temperature)
plt.grid(True)
plt.axis(xmin=0, ymin=0) 

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(impedance['Real(Z)'], impedance['-Im(Z)'].abs())

plt.xlabel('Z\' (ohm)')
plt.ylabel('-Z\'\' (ohm)')
plt.title(temperature)
plt.grid(True)
plt.axis(xmin=0, ymin=0) 


# In[ ]:



