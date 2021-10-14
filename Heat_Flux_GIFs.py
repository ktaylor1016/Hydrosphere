
import numpy as np
import pandas as pd

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

#T0=np.genfromtxt('/Users/karleetaylor/Dropbox/My Mac (Karlee’s MacBook Pro)/Documents/PythonPrograms/T1M1Rb1.txt',usecols=6)
#T0_s=T0[0]

df0=pd.read_csv("T1M1Rb1.txt")




"""
# To read in data with a for loop - this might not actually be possible because can't have for loop-dependent string within file path
#initalize the dictionary 
data = {}
#for loop to get key names
for count1 in range(res_T):
    for count2 in range(res_MW):
        for count3 in range(res_r):
            #key name is what the array is called
            key_name = "Q"+ str(count1 + 1) + str(count2 + 1) + str(count3 + 1)
            #value is what the data actually is
            value = np.genfromtxt('/Users/karleetaylor/Dropbox/My\ Mac\ \(Karlee’s\ MacBook\ Pro\)/Documents/PythonPrograms/"T"+str(count1+1)+"M"+str(count2+1)+"Rb"+str(count3+1)+".txt"',usecols=11)
            #now that we have both we can add this to the dictionary
            data[key_name] = value
"""


"""
for count1 in range(res_T):
    for count2 in range(res_MW):
        for count3 in range(res_r):
            # Q arrays
            array_name = "Q"+str(count1+1)+str(count2+1)+str(count3+1) #name arrays
            array_name = np.genfromtxt('/Users/karleetaylor/Dropbox/My\ Mac\ \(Karlee’s\ MacBook\ Pro\)/Documents/PythonPrograms/__FILE NAME__.txt',usecols=11)
            
            # Q_surface array for all trials
            Q_surface_value = array_name[0] # take first value from array of Q values to get Q_surface
            Q_surface_array=np.append(Q_surface_array,Q_surface_value)

            #Create temp profile arrays
            # z arrays
            array_name2 = "z"+str(count1+1)+str(count2+1)+str(count3+1)
            array_name2 = np.genfromtxt('/Users/karleetaylor/Dropbox/My\ Mac\ \(Karlee’s\ MacBook\ Pro\)/Documents/PythonPrograms/__FILE NAME__.txt',usecols=0)

            # T arrays
            array_name3 = "T"+str(count1+1)+str(count2+1)+str(count3+1)
            array_name3 = np.genfromtxt('/Users/karleetaylor/Dropbox/My\ Mac\ \(Karlee’s\ MacBook\ Pro\)/Documents/PythonPrograms/__FILE NAME__.txt',usecols=6)
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