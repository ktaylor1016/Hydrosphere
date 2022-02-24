# Function that can be used to generate multiple plots of the parameter space of various simulations. Simulations should vary in surface temperature and surface water mass, with all else held constant.

def parameter_space_plots(path, ice_2D=0, ice_3D=0, ice_contour=0, ocean_contour=0, ocean_3D=0):

    # Note: input paths above between quotes ''
    # 1 = plot is ON
    # 0 = plot is OFF

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    import glob
    import shutil
    import matplotlib.cm as cm
    from scipy import interpolate

    # use glob to get all the csv files for the large planet
    csv_files = glob.glob(os.path.join(path, "*.txt"))
    csv_files.sort()

    SurfaceT_array=[]
    HPdepth_array=[]
    Ocean_depth_array = []
    MW_array_kg=[]

    for f in csv_files:

        # find surface T
        df = np.loadtxt(f,delimiter=" ", skiprows=4)
        z=df[:,0]
        z_km=z/1000
        phase=df[:,5]
        T=df[:,6]
        T_surface=T[0]
        SurfaceT_array.append(T_surface)

        # find water mass
        df1 = np.loadtxt(f,delimiter=" ", skiprows=2, max_rows=1)
        print(df1)
        MW=df1[:,2]
        MW_0=MW[0]
        MW_array_kg.append(MW_0)

        # determine depth of HP ice
        hp_surface_i = np.argmax(phase>1) # get index of first phase =>2 (if nothing is found, returns 0)
        if hp_surface_i==0: # replace 0s with bottom of hydrosphere
            hp_surface_i=len(z)-1
        hp_surface_depth=z[hp_surface_i] # find depth at this index
        hp_depth=np.max(z)-hp_surface_depth # subtract this depth from total depth
        HPdepth_array.append(hp_depth)

        # find top and bottom of ocean layer
        ocean_indices = np.where(phase==0)
        ocean_surface_i = np.min(ocean_indices)
        ocean_bottom_i = np.max(ocean_indices)
        ocean_surface_z = z[ocean_surface_i]
        ocean_bottom_z = z[ocean_bottom_i]
        ocean_depth = ocean_bottom_z-ocean_surface_z
        Ocean_depth_array.append(ocean_depth)

    # 2D with colorbar
    if ice_2D==1:
        df = pd.DataFrame(data={'A':SurfaceT_array,'B':MW_array_kg,'C':HPdepth_array})
        points = plt.scatter(df.A, df.B, c=df.C,cmap="cool_r", lw=0)
        cbar = plt.colorbar(points)
        plt.xlabel("Surface Temperature (K)")
        plt.ylabel("Water Mass (%)")
        plt.title("Earth Mass Planet Parameter Space")
        cbar.set_label('High Pressure Ice Thickness (m)', rotation=90)
        plt.savefig('/Users/karleetaylor/Dropbox/My Mac (Karlee’s MacBook Pro)/Documents/PythonPrograms/EM_Parameter_Space.png')
        plt.show()
    
    # 3D interpolated surface - HP ice thickness
    if ice_3D==1:
        df = pd.DataFrame(data={'A':SurfaceT_array,'B':MW_array_kg,'C':HPdepth_array})
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        #ax.scatter(df.A, df.B, df.C)
        ax.plot_trisurf(df.A, df.B, df.C, linewidth=0)
        #ax.set_xlim(260,360,20)
        ax.set_xlabel('Surface Temperature (K)')
        ax.set_ylabel('Water Mass (%)')
        ax.set_zlabel('High Pressure Ice Thickness (m)')
        plt.title("0.8 EM Planet Parameter Space")
        plt.savefig('/Users/karleetaylor/Dropbox/My Mac (Karlee’s MacBook Pro)/Documents/PythonPrograms/3D_0.8EM_Parameter_Space.png')
        plt.show()
    
    # HP Thickness Elevation Map
    if ice_contour==1:
        Temp_array_new = np.linspace(260,350,20)
        HPdepth_2Darray=np.array(HPdepth_array).reshape(20,20) # hp depth must be in 2d array (20 by 20)
        fig, ax = plt.subplots()
        CS = ax.contour(Temp_array_new, MW_array_kg, HPdepth_2Darray/1000) # adjust float later
        ax.clabel(CS, inline=True, fontsize=10)
        ax.set_title('1EM High Pressure Ice Thickness (km)')
        ax.set_xlabel('Surface Temperature (K)')
        ax.set_ylabel('Water Mass (%)')
        plt.savefig('/Users/karleetaylor/Dropbox/My Mac (Karlee’s MacBook Pro)/Documents/PythonPrograms/Contour_1EM_Parameter_Space.png')
        plt.show()

    # Ocean depth elevation map
    if ocean_contour==1:
        Temp_array_new = np.linspace(260,350,20)
        Ocean_depth_2Darray = np.array(Ocean_depth_array).reshape(20,20)
        fig, ax = plt.subplots()
        CS = ax.contour(Temp_array_new, MW_array_kg, Ocean_depth_2Darray/1000) # adjust float later
        ax.clabel(CS, inline=True, fontsize=10)
        ax.set_title('1EM Ocean Depth (km)')
        ax.set_xlabel('Surface Temperature (K)')
        ax.set_ylabel('Water Mass (%)')
        plt.savefig('/Users/karleetaylor/Dropbox/My Mac (Karlee’s MacBook Pro)/Documents/PythonPrograms/Contour_1EM_Ocean_Depth.png')
        plt.show()

    # 3D interpolated surface - ocean depth
    if ocean_3D==1:
        df = pd.DataFrame(data={'A':SurfaceT_array,'B':MW_array_kg,'C':Ocean_depth_array})
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        #ax.scatter(df.A, df.B, df.C)
        ax.plot_trisurf(df.A, df.B, df.C, linewidth=0)
        #ax.set_xlim(260,360,20)
        ax.set_xlabel('Surface Temperature (K)')
        ax.set_ylabel('Water Mass (%)')
        ax.set_zlabel('Ocean Depth (m)')
        plt.title("1 EM Planet Parameter Space")
        plt.savefig('/Users/karleetaylor/Dropbox/My Mac (Karlee’s MacBook Pro)/Documents/PythonPrograms/3D_1EM_Ocean_Depth.png')
        plt.show()

parameter_space_plots('/Users/karleetaylor/Dropbox/My Mac (Karlee’s MacBook Pro)/Documents/PythonPrograms/New_header_files')