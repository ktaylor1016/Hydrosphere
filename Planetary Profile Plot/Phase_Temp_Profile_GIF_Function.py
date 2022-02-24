def phase_temp_profile_gif(path, relative_path):
    # Note: input path between quotes ''

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    import glob
    import shutil

    # use glob to get all the csv files in this folder
    csv_files = glob.glob(os.path.join(path, "*.txt"))


    # specify colors
    import matplotlib
    cmap = matplotlib.cm.get_cmap('cool')
    colors=[]
    for i in range(0,len(csv_files)): # iterate according to number of files
        rgba = cmap(i/len(csv_files)) # get rgba value of color for each file
        colors.append(rgba) # add rgba values to list of colors

    # put in correct order before making plots
    for f in csv_files:
        df = np.loadtxt(f,delimiter=" ", skiprows=4)
        T=df[:,6]
        T_surface=T[0]
        os.rename(f, str(round(T_surface,1))+'.txt')
        filename=str(round(T_surface,1))+'.txt'
        shutil.move(filename, path)
    csv_files.sort()


    # loop over the list of csv files
    for f in csv_files:
        # read the csv file
        df = np.loadtxt(f,delimiter=" ", skiprows=4)
        z=df[:,0]
        z_km=z/1000
        phase=df[:,5]
        T=df[:,6]
        T_surface=T[0]

        fig, (ax1, ax2) = plt.subplots(1,2)
        fig.suptitle('Surface Temperature = '+str(round(T_surface,1))+'K')
        index=csv_files.index(f) # make index correspond to position in sorted temp list above, make list of f in order of surface T... maybe make dict?

        # plot temp profile
        ax1.set_xlabel('Temperature (K)', size=10)
        ax1.set_ylabel('Depth (km)', size=10)
        #ax1.set_ylim([0,max(z)])
        ax1.set_xlim([250,550])
        ax1.invert_yaxis()
        ax1.set_title('Temperature Profile', size=15)
        ax1.tick_params(axis='both', labelsize=10)
        ax1.tick_params(direction='in', length=6, width=2, colors='black')
        ax1.plot(T, z_km, linewidth=2, color=colors[index], label='Temperature = '+str(round(T_surface,1))+'K')

        # plot phase profile
        ax2.set_xlabel('Phase', size=10)
        ax2.set_ylabel('Depth (km)', size=10)
        #ax2.set_ylim([0,max(z)])
        ax2.set_xlim([-0.5,7.5])
        ax2.invert_yaxis()
        ax2.set_title('Phase Profile', size=15)
        ax2.tick_params(axis='both', labelsize=10)
        ax2.tick_params(direction='in', length=6, width=2, colors='black')
        ax2.plot(phase, z_km, linewidth=2, color=colors[index], label='Temperature = '+str(round(T_surface,1))+'K')
        fig.savefig(relative_path+'/'+os.path.basename(str(round(T_surface,1)))+'.png') # RENAME W SURFACE T, SAVE TO SUBFOLDER


    from PIL import Image
    import glob
    

    # Create the frames
    frames = []
    imgs = glob.glob(path+'/*.png')
    imgs.sort(reverse=True) #SORT ACCORDING TO SURFACE T
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)
    
    # Save into a GIF file that loops forever
    frames[0].save('profile_test.gif', format='GIF',
                append_images=frames[1:],
                save_all=True,
                duration=75, loop=0)

    print('done')

phase_temp_profile_gif(path='/Users/karleetaylor/Dropbox/My Mac (Karlee’s MacBook Pro)/Documents/PythonPrograms/New_header_files', relative_path='New_header_files')