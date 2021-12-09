
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob


# use glob to get all the csv files in this folder
path = '/Users/karleetaylor/Dropbox/My Mac (Karlee’s MacBook Pro)/Documents/PythonPrograms/Output Files' #os.getcwd() 
csv_files = glob.glob(os.path.join(path, "*.txt"))
csv_files.sort()
print(csv_files)


# specify colors
import matplotlib
cmap = matplotlib.cm.get_cmap('cool')
colors=[]
for i in range(0,len(csv_files)): # iterate according to number of files
    rgba = cmap(i/len(csv_files)) # get rgba value of color for each file
    colors.append(rgba) # add rgba values to list of colors
print(colors)


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
    index=csv_files.index(f)

    # plot temp profile
    ax1.set_xlabel('Temperature (K)', size=10)
    ax1.set_ylabel('Depth (km)', size=10)
    ax1.set_ylim([0,8000])
    #ax1.set_xlim([250,400]) - could include this to keep axes the same, but then you lose all the detail.
    ax1.invert_yaxis()
    ax1.set_title('Temperature Profile', size=15)
    ax1.tick_params(axis='both', labelsize=10)
    ax1.tick_params(direction='in', length=6, width=2, colors='black')
    ax1.plot(T, z, linewidth=2, color=colors[index], label='Temperature = '+str(round(T_surface,1))+'K')

    # plot phase profile
    ax2.set_xlabel('Phase', size=10)
    ax2.set_ylabel('Depth (km)', size=10)
    ax2.set_ylim([0,8000])
    ax2.set_xlim([-0.5,7])
    ax2.invert_yaxis()
    ax2.set_title('Phase Profile', size=15)
    ax2.tick_params(axis='both', labelsize=10)
    ax2.tick_params(direction='in', length=6, width=2, colors='black')
    ax2.plot(phase, z, linewidth=2, color=colors[index], label='Temperature = '+str(round(T_surface,1))+'K')
    fig.savefig('Output Files/'+os.path.basename(f)+'.png')


from PIL import Image
import glob
 
# Create the frames
frames = []
imgs = glob.glob('/Users/karleetaylor/Dropbox/My Mac (Karlee’s MacBook Pro)/Documents/PythonPrograms/Output Files/*.png')
imgs.sort(reverse=True)
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)
 
# Save into a GIF file that loops forever
frames[0].save('263_to_373K.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=300, loop=0)

print('done')

# go high to low temp
# NEXT - add wedge plot! 
