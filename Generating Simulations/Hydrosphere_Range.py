# Define function w many inputs (HEAT FLUX NOT INCLUDED)

def hydrosphere_range(surface_p, max_surface_t, min_surface_t, res_t, min_water_mass, max_water_mass, res_mw, core_radius, res_radius, core_density, rawdata_flg):

    # Import necessary packages
    import numpy as np
    import seafreeze as sf
    import matplotlib.pyplot as plt
    from tabulate import tabulate
    from scipy.optimize import minimize
 
    # Specify resolution ---- maybe this should be an input?
    res_T = res_t
    res_MW = res_mw
    res_r = res_radius
    tot = res_T*res_MW*res_r
    clo = 0

    # Redefine variables
    P_s_set = [surface_p]
    T_s_set = np.linspace(min_surface_t,max_surface_t,num=res_T)
    Mass_W_i_set = np.linspace(min_water_mass, max_water_mass, num=res_MW) #[water_mass]
    r_b_set = np.linspace(core_radius, core_radius, num=res_r)  #[core_radius]
    rho_core_set = [core_density]

    
    for count1 in range(res_T):
        for count2 in range(res_MW):
            for count3 in range(res_r):
                # data can change arbitrarily
                P_s = 0.1   # Surface Pressure (MPa);
                T_s = T_s_set[count1]   # Surface Temperature (K);
                Mass_W_i = Mass_W_i_set[count2]    # Mass of water in kg;
                r_b = r_b_set[count3]              #1*6370*1e3  # Radius rocky core (m);
                rho_core = 5514    # density of the rocky core (kg/m3);
                clo = clo+1
                ##### Resolution
                #computation time /profile tools 
                res = 100;  # z grid
                Mass_it = 3 # Mass convergence iteration: 3-5 is enough (check % Mass difference) --------------------------------------- this is not used for some reason ------------------------------
                g_it = 2;  # Gravity convergence iteration (3 is enough)
            
                ##### Plots and table
                plot_flg = 0 # plot flag: 0=no plot; 1=plot
                tab_flg = 0 # Table flag: 0=no table; 1=table
                gravity_flg = 0 # gravity profile flag: 0=no plot; 1=plot
                writefile_flg = 0 # Output file flag: 0=no output file; 1=output file
                IO_flg = 0 # Output input/output file flag: 0=no output file; 1=output file
                rawdata_flg = 1 # Output the raw data in columns; 1=output
                piechart_flg = 0 # plot a pie chart: 0=no plot; 1=plot
                scatterplt_flg = 0 # plot a scatter point between P and T: 0=no plot; 1=plot
            
                #### Threshold 
                # Mass covergence loop threshhold, in percentage
                mass_thrshd = 1
                
                #############################################
            
                 # Ice VII approximation
                rho_VII = 1.6169e+03
                alpha_VII = 2.45e-4
                Cp_VII = 3400
            
                # Ice Ih conductivity (Andersson and Akira, 2005) 
                def K_Ih(P,T):
                    return -2.04104873*P+(632/T+0.38-0.00197*T)
            
                # Rocky/metal core calculation
                Mass_core = 4/3*np.pi*r_b**3*rho_core
                g_s = 6.67430e-11*Mass_core/r_b**2 # Gravity at the Hydrosphere Mantle Boundary
                depth = (Mass_W_i/800/(4/3*np.pi)+r_b**3)**(1/3)-r_b # Depth approximation in m
                wtpW = Mass_W_i/(Mass_core+Mass_W_i)*100
                # Testing factors
                massDiff = 100
                EnableTest = 0 #0=disable, 1=enable
            
            
                # initializing the grids
                z = np.linspace(0, depth, num=res)  # depth grid
                rho = np.zeros(z.size)  # Density grid
                alpha = np.zeros(z.size)  # thermal epansivity grid
                Cp = np.zeros(z.size)  # Heat capacity grid
                dT_dz = np.zeros(z.size)  # thermal gradient grid
                phase = np.zeros(z.size)  # phase grid
                T = np.zeros(z.size)  # Temperature grid
                P = np.zeros(z.size)  # Pressure grid
                grav = np.zeros(z.size) # gravity grid
                M_L = np.zeros(z.size) # Mass grid

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
                        dT_dz_s = alpha_s*g[0]*T_s/Cp_s  # Thermal gradient at the surface  USE THIS
                        T[0] = T_s
                        P[0] = P_s
                        rho[0] = rho_s
                        alpha[0] = alpha_s
                        Cp[0] = Cp_s
                        dT_dz[0] = dT_dz_s
                        phase[0] = phase_s[0]


                        for i in range(z.size-1):  # Integration with depth
                            T[i+1] = T[i] + dT_dz[i] * (z[i+1]-z[i]);
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
                            dT_dz[i+1] = alpha[i+1]*g[i+1]*T[i+1]/Cp[i+1];

                        # Gravity in the hydrosphere
                        for i in range(1,len(rho)):
                            M_L[i]=rho[i]*4/3*np.pi*((r_b+z[i-1]+(depth/res))**3-(r_b+z[i-1])**3)
                        Mass_Shells = np.cumsum(np.flip(M_L,0))

                        for i in range(len(rho)):    
                            grav[i] = 6.67430e-11*(Mass_core+Mass_Shells[i])/(r_b+z[i])**2  

                    # Compute Mass
                    Mass_WL = np.sum(M_L)
                    Mass_diff = Mass_W_i-Mass_WL

                    # depth difference for Mass convergence
                    depth_diff = (np.abs(Mass_diff)/(np.mean(rho)*1.8)/(4/3*np.pi)+r_b**3)**(1/3)-r_b
                    if   Mass_diff > 0:  
                        depth = depth + depth_diff*0.8
                    else:
                        depth = depth - depth_diff*0.8

                    if EnableTest == 1:
                        print("depth after " + str(depth))
                        print()
                    print('Mass Convergence (%): ' + str(Mass_diff/Mass_WL*100))
                
                # Compute Mass
                E_M_WL = Mass_WL/5.9722e24 # Mass Water Layer in Earth mass
                O_M_WL = Mass_WL/1.4e21 # Mass Water Layer in Ocean mass (Earth) - this is never used, so not a problem

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
                    filename = "T"+str(count1+1).zfill(3)+"M"+str(count2+1).zfill(3)+"Rb"+str(count3+1).zfill(3)+".txt"
                    with open(filename, mode = "w") as file0:
                        
                        file0.write("INPUTS\n") # ADD INPUTS HERE
                        file0.write("Surface_Temperature(K) Surface_Pressure(MPa) Water_Mass(kg) Rocky_Core_Radius(m) Rocky_Core_Density(kg/m3)\n") 
                        file0.write(str(T_s)+" "+str(surface_p)+" "+str(Mass_W_i)+" "+str(core_radius)+" "+str(core_density)+"\n")

                        file0.write("depth density thermal_expansivity heat_capacity thermal_gradient ")
                        file0.write("phase temperature pressure gravity mass\n")
                        for i in range(rho.size):
                            file0.write(str(z[i])+" "+str(rho[i])+" "+str(alpha[i])+" "+str(Cp[i])+" "+str(dT_dz[i])+" ")
                            file0.write(str(phase[i])+" "+str(T[i])+" "+str(P[i])+" "+str(grav[i])+" "+str(M_L[i])+"\n")

# Calculate data for T=260-350 K and MW=1-20% total mass, 0.8EM
#hydrosphere_range(surface_p=0.1, min_surface_t=260, max_surface_t=350, res_t=20, min_water_mass=0.8*0.01*5.972e+24, max_water_mass=0.8*0.20*5.972e+24, res_mw=20, core_radius=0.8*0.6*6.36e+06, res_radius=1, core_density=5500,rawdata_flg=1)
hydrosphere_range(surface_p=0.1, min_surface_t=340, max_surface_t=350, res_t=2, min_water_mass=0.8*0.01*5.972e+24, max_water_mass=0.8*0.20*5.972e+24, res_mw=2, core_radius=0.8*0.6*6.36e+06, res_radius=1, core_density=5500,rawdata_flg=1)

# sims for plot generator
#hydrosphere_range(surface_p=0.1, min_surface_t=250, max_surface_t=350, res_t=5, min_water_mass=0.01*5.972e+24, max_water_mass=0.20*5.972e+24, res_mw=5, core_radius=0.6*6.36e+06, res_radius=1, core_density=5500,rawdata_flg=1)

# not necessarily getting stuck, just that mass convergence values start decreasing very slowly. Always happens for largest water mass. Should only iterate 5 times, since that's what is in script?
# tried changing mass covergence iteration #, but didn't fix. For some reason, this parameter is not actually used in script.
# or try changing mass_thrshd ------ ORIGINALLY AT 1, CHANGED TO 10 AND SEEMS TO BE FIXED!
print('done')
