
import numpy as np

#Test values from Titan
#surface_p=0.1 (MPa)
#surface_t=263
#water_mass=1.4e+21
#core_radius=0.6*6.36e+06
#core_density=5500
#target_planet=2

def hydrosphere_flux(surface_p, max_surface_t, min_surface_t, res_t, water_mass, res_mw, core_radius, res_radius, core_density, rawdata_flg):

    # Import necessary packages
    from scipy.optimize import minimize
    from tabulate import tabulate
    import matplotlib.pyplot as plt
    import seafreeze as sf
    import numpy as np

    # Specify resolution
    res_T = res_t
    res_MW = res_mw
    res_r = res_radius
    tot = res_T*res_MW*res_r
    clo = 0

    # Redefine variables
    P_s_set = [surface_p]
    T_s_set = np.linspace(min_surface_t,max_surface_t,num=res_T)
    Mass_W_i_set = np.linspace(water_mass, water_mass, num=res_MW) #[water_mass]
    r_b_set = np.linspace(core_radius, core_radius, num=res_r)  #[core_radius]
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

                    
                    # Compute Thermal Conducitiviy & Heat Flux                                        ################### Computing Heat Flux ###############
                    def K_Ih(i):
                        return 2.2207*(1+0.105*(T[i]-273.15))  #from Wikipedia, 2.2207(1+0.105(T-273.15)), Units are W/m*K

                    def q_Ih(i):
                        return -K_Ih(i)*dT_dz[i] #From Planet Profile: -K_Ih*np.log(T[i]/T_s)/(z[i+1]-z[i]), Units are W/m*K

                    def K_water(i):
                        return -8.354*(2.71828)-6*((T[i])**2)+6.53*2.71828-3*T[i]-0.5981 ## Units unknown, from https://www.researchgate.net/post/Thermal_conductivity_of_water

                    def q_water(i):
                        return -K_water(i)*dT_dz[i] ## CHECK UNITS

                    # Compute Surface Heat Flux & Internal Heating
                    for i in range(0,1):
                        if phase[i]==0:
                            surface_heat_flux=q_water(i)
                            surface_internal_heating=surface_heat_flux*4*3.14159*(z[i+1])**2
                            print("Surface heat flux is "+str(surface_heat_flux)+" W/m^2")
                            print("Surface internal heating is "+str(surface_internal_heating)+" W")
                        if phase[i]==1:
                            surface_heat_flux=q_Ih(i)
                            surface_internal_heating=surface_heat_flux*4*3.14159*(z[i+1])**2
                            print("Surface heat flux is "+str(surface_heat_flux)+" W/m^2")
                            print("Surface internal heating is "+str(surface_internal_heating)+" W")

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
                        file0.write("depth density thermal_expansivity heat_capacity thermal_gradient phase temperature pressure gravity mass surface_heat_flux total_internal_heat\n")
                        for i in range(rho.size):
                            file0.write(str(z[i])+" "+str(rho[i])+" "+str(alpha[i])+" "+str(Cp[i])+" "+str(dT_dz[i])+" "+str(phase[i])+" "+str(T[i])+" "+str(P[i])+" "+str(grav[i])+" "+str(M_L[i])+" "+str(surface_heat_flux)+" "+str(surface_internal_heating)+"\n")

# Calculate heat flux for values of surface temperature between 263 and 373
hydrosphere_flux(surface_p=0.1, min_surface_t=263, max_surface_t=373, res_t=10, water_mass=1.4e+21, res_mw=1, core_radius=0.6*6.36e+06, res_radius=1, core_density=5500,rawdata_flg=1)


"""
# original raw data output code:
                #print(count3)
                # Output the raw data to seperated files 
                if rawdata_flg == 1:
                    filename = "T"+str(count1+1)+"M"+str(count2+1)+"Rb"+str(count3+1)+".txt"
                    with open(filename, mode = "w") as file0:
                        file0.write("depth density thermal_expansivity heat_capacity thermal_gradient ")
                        file0.write("phase temperature pressure gravity mass\n surface_heat_flux total_internal_heat")
                        for i in range(rho.size):
                            file0.write(str(z[i])+" "+str(rho[i])+" "+str(alpha[i])+" "+str(Cp[i])+" "+str(dT_dz[i])+" ")
                            file0.write(str(phase[i])+" "+str(T[i])+" "+str(P[i])+" "+str(grav[i])+" "+str(M_L[i])+"\n+ "+str(surface_heat_flux)+" "+str(surface_internal_heating))
"""