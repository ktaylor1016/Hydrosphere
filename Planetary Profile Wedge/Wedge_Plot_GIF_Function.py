# Create a GIF of planetary evolution seen through a planetary 'wedge'. Data files used should have a range of temperatures, with all other variables constant.

def wedge_plot_gif(path, relative_path):

    # Note: Input path to data files using quotes ''

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    import glob
    import shutil
    from matplotlib.patches import Wedge
    from matplotlib.collections import PatchCollection
    import matplotlib.colors as mcolors

    # Get CSV files
    csv_files = glob.glob(os.path.join(path, "*.txt")) # Extract the text files from this folder

    # Sort files according to temp (high to low)
    for f in csv_files:
        df = np.loadtxt(f,delimiter=" ", skiprows=6)
        T=df[:,6]
        T_surface=T[0]
        os.rename(f, str(round(T_surface,1))+'.txt')
        filename=str(round(T_surface,1))+'.txt'
        shutil.move(filename, path) # Return files to path, now sorted correctly
    csv_files.sort()


    # Loop over list of CSV files
    for f in csv_files:
        # Read in files
        df = np.loadtxt(f,delimiter=" ", skiprows=4)
        z=df[:,0]
        phase=df[:,5]
        T=df[:,6]
        T_surface=T[0]

        # Find location of phase changes
        from numpy import diff
        # Find derivatives of graph
        x = z
        y = phase
        dydx = diff(y)/diff(x)
        # Find index of phase change pts (locations where slope does not equal 0)
        inflection_pts = np.where(dydx != 0) # Index of phase change
        # Find z value (depth) that corresponds to this index
        phase_change = [] # Depth of phase change
        for i in inflection_pts:
            inflection_depth = z[i]
            phase_change.append(inflection_depth)  
        phase_change = np.append(phase_change, z[0]) # Add top of hydrosphere
        phase_change = np.sort(phase_change) # List in ascending order

        # Find phase between inflection points (at 1 depth step below location of each change)
        phase_numbers=[]
        for i in range(len(phase_change)):
            index = np.where(z == phase_change[i]) # Get index of depth of phase change
            index2 = index[0]
            phase_number = phase[index2+1] # Get phase at one step below change
            phase_numbers.append(phase_number) # create list of keys with phase #

        # Add bottom of hydrosphere
        max = len(z)-1
        phase_change = np.append(phase_change, z[max]) # Depth
        phase_numbers.append(100) # "Phase" (just for matching colors)
        
        # Starting point for each wedge
        x = [0]*len(phase_numbers)
        y = [0]*len(phase_numbers)
        
        # THICKNESS OF LAYERS
        r_b=6360000 #radius of core
        # Radius of each wedge (thickness of layers)
        radius=[]
        for i in range(len(phase_change)):
            radius_i = (phase_change[0]-phase_change[i]+r_b)/1E6
            radius.append(radius_i)
        
        # Start and end angle of the wedge
        start_angle = [75]*len(phase_numbers)
        end_angle = [105]*len(phase_numbers)

        # List of colors (unique to each file)
        colors_dict = {0: "navy", 1: "aqua", 6: "mediumorchid", 7: "mediumvioletred", 100: "saddlebrown"}
        colors = []
        for i in phase_numbers:
            int_i=int(i)
            phase_color = colors_dict.get(int_i)
            colors.append(phase_color)

        # Graph
        patches = []
        for x1, y1, r, t1, t2 in zip(x, y, radius, start_angle, end_angle):
            wedge = Wedge((x1, y1), r, t1, t2)
            patches.append(wedge)

        # Label
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color="navy", lw=3),
                    Line2D([0], [0], color="aqua", lw=3),
                    Line2D([0], [0], color="mediumorchid", lw=3),
                    Line2D([0], [0], color="mediumvioletred", lw=3),
                    Line2D([0], [0], color="saddlebrown", lw=3)]

        fig, ax = plt.subplots()
        p = PatchCollection(patches, alpha=0.8, color=colors)
        ax.add_collection(p)
        ax.set_xlim(-2,2)
        ax.set_ylim(5.0,6.5)
        ax.set_title('Surface Temperature = '+str(round(T_surface,1))+' K')
        ax.legend(custom_lines, ['Water', 'Ice Ih', 'Ice VI', 'Ice VII', 'Silicate Core'], loc='lower right')
        plt.ylabel('Depth (1000 km)')
        fig.suptitle('Planetary Profile')

        # Note: one may choose to add/delete/adjust colors and legend names included in GIF. Since Water, Ice Ih, Ice VI, and Ice VII are the most common phases present in simulations, those are the only phases listed to prevent the image from being too crowded. 
            # To delete phases, simply delete the corresponding pieces of code at lines 84, 99, and 111.
            # To add phases, add a color and phase to the colors dictionary (line 84), add a line of code after line 99, and add the phase to the legend in line 111.

        fig.savefig(relative_path+'/'+os.path.basename(str(round(T_surface,1)))+'_loop_wedge.png') # Adjust this line to be the location of where the frames of the GIF will be saved 

    
    # Create the frames of the GIF
    from PIL import Image
    import glob
    frames = []
    imgs = glob.glob(path+'/*.png') # The path here should be the location from above, where the GIF frames are located
    imgs.sort(reverse=True) # Sort according to surface T
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)
    
    # Save into a GIF
    # Adjust the name, duration (speed), and # of loops (0 = repeats infinitely) of the GIF below
    frames[0].save('Wedge_loop_test.gif', format='GIF',
                append_images=frames[1:],
                save_all=True,
                duration=100, loop=0)

    print('Done')

wedge_plot_gif(path='/Users/karleetaylor/Dropbox/My Mac (Karlee’s MacBook Pro)/Documents/PythonPrograms/New_header_files', relative_path='New_header_files')