
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Text files should save to PythonPrograms folder

# Open and read in text files, extract surface Q values
#maybe use pandas to read in data files more easily

res_T = 10
res_MW=1
res_r=1
min_surface_t=263
max_surface_t=373
surface_t_array = np.linspace(min_surface_t,max_surface_t,num=res_T)

T_surface_array = []
Q_surface_array = []

# read in raw data files
import pandas as pd
import os
import glob
  
# use glob to get all the csv files in this folder
path = '/Users/karleetaylor/Dropbox/My Mac (Karlee’s MacBook Pro)/Documents/PythonPrograms/Output Files' #os.getcwd() 
csv_files = glob.glob(os.path.join(path, "*.txt"))
print(csv_files) #empty

#os.listdir(path)
#files=os.listdir(path)
#for f in myFiles:
    #data=np.loadtxt(f,delimeter=" ", skiprows=6)
#myFiles=glob.glob('*.txt)

# loop over the list of csv files
for f in csv_files:
    # read the csv file
    df = np.loadtxt(f,delimiter=" ", skiprows=6)
    z=df[:,0]
    phase=df[:,5]
    T=df[:,6]
    T_surface=T[0]

    #fig, axs = plt.subplots(2)
    #fig.suptitle('Temperature = '+str(round(T_surface,1))+'K')

    # plot temp profile
    fig, ax = plt.subplots(figsize=(15, 11))
    ax.set_xlabel('Temperature (K)', size=30)
    ax.set_ylabel('Depth (km)', size=30)
    ax.invert_yaxis()
    ax.set_title('Temperature Profile', size=25)
    ax.tick_params(axis='both', labelsize=25)
    ax.tick_params(direction='in', length=6, width=2, colors='black')
    ax.plot(T, z, linewidth=2, color='blue', label='Temperature = '+str(round(T_surface,1))+'K')
    plt.legend(loc="upper right", prop={'size':20})
    #plt.show()
    fig.savefig('Output Files/Figures/temp_profile.png') # change f to name of file

    # plot phase profile
    fig, ax = plt.subplots(figsize=(15, 11))
    ax.set_xlabel('Phase', size=30)
    ax.set_ylabel('Depth (km)', size=30)
    plt.xlim([-0.5,7])
    ax.invert_yaxis()
    ax.set_title('Phase Profile', size=25)
    ax.tick_params(axis='both', labelsize=25)
    ax.tick_params(direction='in', length=6, width=2, colors='black')
    ax.plot(phase, z, linewidth=2, color='blue', label='Temperature = '+str(round(T_surface,1))+'K')
    plt.legend(loc="upper right", prop={'size':20})
    fig.savefig('Output Files/Figures/phase_profile.png')



"""
# make for loop for reading in files?
df0=pd.read_csv('/Users/karleetaylor/Dropbox/My Mac (Karlee’s MacBook Pro)/Documents/PythonPrograms/Output Files/T1M1Rb1.txt', usecols=[0,5,6,10,11], sep=' ') 
z0=df0["depth"]
phase0=df0["phase"]
T0=df0["temperature"]
q0=df0["surface_heat_flux"]
Q0=df0["total_internal_heat"]

df1=pd.read_csv('/Users/karleetaylor/Dropbox/My Mac (Karlee’s MacBook Pro)/Documents/PythonPrograms/Output Files/T2M1Rb1.txt', usecols=[0,5,6,10,11], sep=' ') 
z1=df1["depth"]
phase1=df1["phase"]
T1=df1["temperature"]
q1=df1["surface_heat_flux"]
Q1=df1["total_internal_heat"]


# plot temp profile
fig, ax = plt.subplots(figsize=(15, 11))
ax.set_xlabel('Depth (km)', size=30)
ax.set_ylabel('Temperature (K)', size=30)
ax.set_title('Temperature Profile', size=25)
ax.tick_params(axis='both', labelsize=25)
ax.tick_params(direction='in', length=6, width=2, colors='black')
ax.plot(z0, T0, linewidth=2, color='blue')
plt.show()

# plot phase profile
fig, ax = plt.subplots(figsize=(15, 11))
ax.set_xlabel('Depth (km)', size=30)
ax.set_ylabel('Phase', size=30)
ax.set_title('Phase Profile', size=25)
ax.tick_params(axis='both', labelsize=25)
ax.tick_params(direction='in', length=6, width=2, colors='black')
ax.plot(z0, phase0, linewidth=2, color='blue')
plt.show()

# make single values, append to array for diff temps
#Q0 = access single cell https://stackoverflow.com/questions/16729574/how-to-get-a-value-from-a-cell-of-a-dataframe
"""

"""
# Plot surface flux vs surface temp
fig, ax = plt.subplots(figsize=(15, 11))
ax.set_xlabel('Surface Temperature (K)', size=30)
ax.set_ylabel('Internal Heating (Q) at Surface (TW)', size=30)
ax.set_title('Internal Heating as a Function of Surface Temperature', size=25)
ax.tick_params(axis='both', labelsize=25)
ax.tick_params(direction='in', length=6, width=2, colors='black')
ax.plot(surface_t_array, Q_surface_array, linewidth=2, color='blue')

# to make GIF of temperature profiles:
#https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python
#https://pypi.org/project/celluloid/

from celluloid import Camera
"""

print('done')