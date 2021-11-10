
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

    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.suptitle('Temperature = '+str(round(T_surface,1))+'K')

    # plot temp profile
    #fig, ax = plt.subplots(figsize=(15, 11))
    ax1.set_xlabel('Temperature (K)', size=10)
    ax1.set_ylabel('Depth (km)', size=10)
    plt.ylim([0,max(z)])
    ax1.invert_yaxis()
    ax1.set_title('Temperature Profile', size=15)
    ax1.tick_params(axis='both', labelsize=10)
    ax1.tick_params(direction='in', length=6, width=2, colors='black')
    ax1.plot(T, z, linewidth=2, color='blue', label='Temperature = '+str(round(T_surface,1))+'K')
    #plt.legend(loc="upper right", prop={'size':20})
    #plt.show()
    #fig.savefig('Output Files/Figures/temp_profile.png') # change f to name of file

    # plot phase profile
    #fig, ax = plt.subplots(figsize=(15, 11))
    ax2.set_xlabel('Phase', size=10)
    ax2.set_ylabel('Depth (km)', size=10)
    plt.ylim([0,max(z)])
    plt.xlim([-0.5,7])
    ax2.invert_yaxis()
    ax2.set_title('Phase Profile', size=15)
    ax2.tick_params(axis='both', labelsize=10)
    ax2.tick_params(direction='in', length=6, width=2, colors='black')
    ax2.plot(phase, z, linewidth=2, color='blue', label='Temperature = '+str(round(T_surface,1))+'K')
    #plt.legend(loc="upper right", prop={'size':20})
    fig.savefig('Output Files/'+os.path.basename(f)+'.png')

from PIL import Image
import glob
 
# Create the frames
frames = []
imgs = glob.glob("*.png")
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)
 
# Save into a GIF file that loops forever
frames[0].save('png_to_gif.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=300, loop=0)

"""
# to make GIF of temperature profiles:
#https://pythonprogramming.altervista.org/png-to-gif/
#https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python
#https://pypi.org/project/celluloid/

from celluloid import Camera
"""

print('done')