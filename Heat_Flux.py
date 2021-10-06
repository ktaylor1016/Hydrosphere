
import numpy as np

#Test values from Titan
#surface_p=0.1 (MPa)
#surface_t=263
#water_mass=1.4e+21
#core_radius=0.6*6.36e+06
#core_density=5500
#target_planet=2

def hydrosphere_flux(surface_p, max_surface_t, min_surface_t, res_t, water_mass, core_radius, core_density):

    # Import necessary packages
    from scipy.optimize import minimize
    from tabulate import tabulate
    import matplotlib.pyplot as plt
    import seafreeze as sf
    import numpy as np

    # Specify resolution
    res_T = res_t
    res_MW = 10
    res_r = 5
    tot = res_T*res_MW*res_r
    clo = 0

    # Redefine variables
    P_s_set = [surface_p]
    T_s_set = np.linspace(min_surface_t,max_surface_t,num=res_T)
    Mass_W_i_set = [water_mass]
    r_b_set = [core_radius]
    rho_core_set = [core_density]

    for count1 in range(res_T):
        for count2 in range(res_MW):
            for count3 in range(res_r):
                # data can change arbitrarily
                P_s = 0.1   # Surface Pressure (MPa);
                T_s = T_s_set[count1]   # Surface Temperature (K);
                Mass_W_i = Mass_W_i_set[count2]    # Mass of water in kg;
                r_b = r_b_set[count3]  # 1*6370*1e3  # Radius rocky core (m);
                rho_core = 5514    # density of the rocky core (kg/m3);
                clo = clo+1
                # Resolution
                # computation time /profile tools
                res = 100  # z grid
                # Mass convergence iteration: 3-5 is enough (check % Mass difference)
                Mass_it = 5
                g_it = 2  # Gravity convergence iteration (3 is enough)

                # Threshold
                # Mass covergence loop threshhold, in percentage
                mass_thrshd = 1

                #############################################

                # Ice VII approximation
                rho_VII = 1.6169e+03
                alpha_VII = 2.45e-4
                Cp_VII = 3400

                # Ice Ih conductivity (Andersson and Akira, 2005)    ######################## Define thermal conductivity #######################
                def K_Ih(P, T):
                    return -2.04104873*P+(632/T+0.38-0.00197*T)

                # Rocky/metal core calculation
                Mass_core = 4/3*np.pi*r_b**3*rho_core
                g_s = 6.67430e-11*Mass_core/r_b**2  # Gravity at the Hydrosphere Mantle Boundary
                depth = (Mass_W_i/800/(4/3*np.pi)+r_b**3)**(1/3) - \
                    r_b  # Depth approximation in m
                wtpW = Mass_W_i/(Mass_core+Mass_W_i)*100
                # Testing factors
                massDiff = 100
                EnableTest = 0  # 0=disable, 1=enable

                # initializing the grids
                z = np.linspace(0, depth, num=res)  # depth grid
                rho = np.zeros(z.size)  # Density grid
                alpha = np.zeros(z.size)  # thermal epansivity grid
                Cp = np.zeros(z.size)  # Heat capacity grid
                dT_dz = np.zeros(z.size)  # thermal gradient grid
                phase = np.zeros(z.size)  # phase grid
                T = np.zeros(z.size)  # Temperature grid
                P = np.zeros(z.size)  # Pressure grid
                grav = np.zeros(z.size)  # gravity grid
                M_L = np.zeros(z.size)  # Mass grid
                

                Mass_WL = 0
                print('-----=== Run ' + str(clo) + '/' + str(tot) + ', ' + str(T_s) + 'K, ' + str(wtpW) + ' wt%, '  + str(r_b*1e-3) + ' km radius' + ' ===-----')
                while (massDiff > mass_thrshd):
                    if EnableTest == 1:
                        print("depth before " + str(depth))
                        
                    # For mass loop the factor being iterated is /depth/

                    # initializing the grids
                    z = np.linspace(0, depth, num=res)  # depth grid

                    grav[:]=g_s # Constant gravity to start with ## set all elements to g_s

                    massDiff = np.abs(100*(Mass_W_i-Mass_WL)/Mass_W_i)
                    #print(massDiff)

                    # Gravity conversion loop
                    for k in range(g_it) if (massDiff==100 or massDiff<mass_thrshd) else range(1): 

                        # For gravity loop the factor being iterated is /grav/

                        g = np.flip(grav,0)
                        PT = np.empty((1,), np.object)
                        PT[0] = (P_s, T_s)
                        #phase_ssolution = sf.whichphase(PT)  # not necessary 
                        if P_s > 2200:
                            out.rho = rho_VII
                            out.alpha = alpha_VII
                            out.Cp = Cp_VII
                            phase_s[0] = 7
                        else:
                            phase_s = sf.whichphase(PT)
                            out = sf.seafreeze(PT,sf.phasenum2phase[phase_s[0]]) 

                        rho_s = out.rho  # Density at the surface
                        alpha_s = out.alpha  # Thermal expansivity at the surface
                        Cp_s = out.Cp  # Heat capacity at the surface
                        dT_dz_s = alpha_s*g[0]*T_s/Cp_s  # Thermal gradient at the surface  ############ Defining temp gradient at surface ##############
                        T[0] = T_s
                        P[0] = P_s
                        rho[0] = rho_s
                        alpha[0] = alpha_s
                        Cp[0] = Cp_s
                        dT_dz[0] = dT_dz_s
                        phase[0] = phase_s[0]


                        for i in range(z.size-1):  # Integration with depth
                            T[i+1] = T[i] + dT_dz[i] * (z[i+1]-z[i]);             ########## Filling in temp grid ############
                            P[i+1] = P[i] + rho[i] * g[i] * (z[i+1]-z[i])*1e-6;
                            PT[0] = (P[i+1],T[i+1])
                            if P[i+1] > 2200:
                                Tm=((P[i+1]*1e-3-2.17)/1.253+1)**(1/3)*354.8
                                if T[i+1] < Tm:
                                    out.rho = rho_VII
                                    out.alpha = alpha_VII
                                    out.Cp = Cp_VII
                                    phase[i+1] = 7
                                else:
                                    phase[i+1] = 0
                                    out = sf.seafreeze(PT, 'water2')

                                    #print('water 2 is used!')

                            else:
                                phase[i+1] = sf.whichphase(PT)
                                out = sf.seafreeze(PT,sf.phasenum2phase[phase[i]])
                            rho[i+1] = out.rho;
                            alpha[i+1] = out.alpha;
                            Cp[i+1] = out.Cp;
                            dT_dz[i+1] = alpha[i+1]*g[i+1]*T[i+1]/Cp[i+1];    ################# Filling in temp gradient grid ####################

                        # Gravity in the hydrosphere
                        for i in range(1,len(rho)):
                            M_L[i]=rho[i]*4/3*np.pi*((r_b+z[i-1]+(depth/res))**3-(r_b+z[i-1])**3)
                        Mass_Shells = np.cumsum(np.flip(M_L,0))

                        for i in range(len(rho)):    
                            grav[i] = 6.67430e-11*(Mass_core+Mass_Shells[i])/(r_b+z[i])**2  

                    # Compute Heat Flux                                        ################### Computing Heat Flux ###############
                    D_Ih=632    # from Andersson and Inaba 2005
                    def q_Ih(i):
                        return D_Ih*np.log(T[i]/T_s)/(z[i+1]-z[i]) ## CHECK UNITS

                    def k_water(i):
                        return 0.00565*[1+0.00319*(T[i]+273.15)-0.0000103*(T[i]+273.15)**2] ## CHECK UNITS - from Riedel 1949, gives thermal conductivity in watt/cm °C). 
                    def q_water(i):
                        return -k_water[i]*dT_dz[i] ## CHECK UNITS

                    # Find depth of bottom of surface layer
                    from numpy import diff
                    # can use diff(y)/diff(x) like I did in wedge plot, but this has same problem of having to iterate through all i to calculate derivative and find inflection pts

                    q_grid = np.zeros(z.size) # Heat flux grid
                    for i in range(z.size-1):                   ########## CHANGE - CALCULATE *JUST* FOR SURFACE LAYER? ############ for all i in phase == 1: ..., for all i in phase == 0: ... just calculate for i = 0 and max i
                        if phase[i]==1:
                            np.append(q_grid, q_Ih(i))
                        if phase[i]==0:
                            np.append(q_grid, q_water(i))

                    def Q(i): # heat production, total heat
                        return q_grid[i]*4*3.14159*(z[i])**2
                    Q_grid = np.zeros(z.size)       # Internal Heating grid
                    for i in range(z.size-1):
                        np.append(Q_grid,Q(i))
                    #units should be TW - 10-100s

                # Compute Internal Heating
                Surface_Internal_Heating = Q[0]

                # Compute Mass
                Mass_WL = np.sum(M_L)
                Mass_diff = Mass_W_i-Mass_WL

                # depth difference for Mass convergence
                depth_diff = (np.abs(Mass_diff)/(np.mean(rho)*1.8)/(4/3*np.pi)+r_b**3)**(1/3)-r_b
                if   Mass_diff > 0:  
                        depth = depth + depth_diff
                else:
                    depth = depth - depth_diff

                if EnableTest == 1:
                    print("depth after " + str(depth))
                    print()
                print('Mass Convergence (%): ' + str(Mass_diff/Mass_WL*100))
                    
                # Compute Mass
                E_M_WL = Mass_WL/5.9722e24 # Mass Water Layer in Earth mass
                O_M_WL = Mass_WL/1.4e21 # Mass Water Layer in Ocean mass (Earth)

                # Boundary of each layer and their thickness
                bd = []
                phase_diff = phase[0]
                count = 1
                phasenum = 1
                phasediffstat = []
                phasediffstat.append(phase_diff)
                for i in range(phase.size-1):
                    if phase_diff == phase[i+1]:
                        count += 1
                    else:
                        bd.append(count)
                        phase_diff = phase[i+1]
                        count = 1
                        phasenum += 1
                        phasediffstat.append(phase_diff)
                bd.append(count)
                boundary = [[0, bd[0]-1]]
                for i in range(len(bd)-1):
                    boundary.append([boundary[i][1],bd[i+1]-1+boundary[i][1]+1])

                sumdepth = [0]
                for i in range(len(boundary)):
                    a = boundary[i][0]
                    b = boundary[i][1]
                    print('Layer '+str(i+1)+' is in phase '+str(int(phase[a])),\
                            ' of depth from '+str(z[a])+'km to '+str(z[b])+'km')
                    sumdepth.append(z[b])
                    print('Profile done!')
                sumdepth.append(z[b]+r_b)


                #print(count3)
                # Output the raw data to seperated files 
                if rawdata_flg == 1:
                    filename = "T"+str(count1+1)+"M"+str(count2+1)+"Rb"+str(count3+1)+".txt"
                    with open(filename, mode = "w") as file0:
                        file0.write("depth density thermal_expansivity heat_capacity thermal_gradient ")
                        file0.write("phase temperature pressure gravity mass\n heat_flux total_internal_heat")
                        for i in range(rho.size):
                            file0.write(str(z[i])+" "+str(rho[i])+" "+str(alpha[i])+" "+str(Cp[i])+" "+str(dT_dz[i])+" ")
                            file0.write(str(phase[i])+" "+str(T[i])+" "+str(P[i])+" "+str(grav[i])+" "+str(M_L[i])+"\n+ "+str(q_grid[i])+" "+str(Q_grid[i]))  ### Think I added heat flux here?

# Calculate heat flux for values of surface temperature between 263 and 373
hydrosphere_flux(surface_p=0.1, min_surface_t=263, max_surface_t=373, res_t=10, water_mass=1.4e+21,core_radius=0.6*6.36e+06,core_density=5500)

# Text files should save to PythonPrograms folder

# Open and read in text files, extract surface Q values

"""
res_T = 10
res_MW = 10
res_r = 5

min_surface_t=263
max_surface_t=373
surface_t_array = np.linspace(min_surface_t,max_surface_t,num=res_T)

n = 10 #number of different surface_t's
Q_surface_array = np.empty(shape=[1,n])

#maybe use pandas to read in data files more easily

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

"""
print('done')