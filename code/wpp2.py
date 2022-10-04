import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import timeit

def symbolic_matrices(n, core = True):

    #define the angle parameters
    beta1 = sp.Symbol('beta_1', Real = True)
    beta2 = sp.Symbol('beta_2', Real = True)
    
    #define the random coefficients based on the imposed symmetries
    #I think the scale does not really matter here. But maybe it does I am not sure actually
    aa = np.random.rand(n)-1/2
    cc = np.random.rand(n)-1/2
    dd = np.random.rand(2 * n).reshape((2,n)) - 1/2
    
    #define the functional forms of x and z
    x = 0
    z = 0
    
    #iterate through the harmonics present in the system
    for i in range(n):
        #add the corresponding terms to the x and z components
        x += aa[i] * sp.sin( (i+1) * beta2)
        z += cc[i] * sp.sin( (i+1) * beta1) + dd[0,i] * sp.cos( (i+1) * beta1) + dd[1,i] * sp.cos( (i+1) * beta2)
    
    #create the symbolic derivatives of the components
    d1_x = sp.diff(x, beta1)
    d1_z = sp.diff(z, beta1)
    d2_x = sp.diff(x, beta2)
    d2_z = sp.diff(z, beta2)
    
    #make lamba functions out of these 
    xf = sp.utilities.lambdify(['beta_1', 'beta_2'], x)
    zf = sp.utilities.lambdify(['beta_1', 'beta_2'], z)
    d1_xf = sp.utilities.lambdify(['beta_1', 'beta_2'], d1_x)
    d1_zf = sp.utilities.lambdify(['beta_1', 'beta_2'], d1_z)
    d2_xf = sp.utilities.lambdify(['beta_1', 'beta_2'], d2_x)
    d2_zf = sp.utilities.lambdify(['beta_1', 'beta_2'], d2_z)

    if core:
        #and return the lambdifyied expressions and also the symbols
        return xf, zf, d1_xf, d1_zf, d2_xf, d2_zf
    
    else:
        return x, z, xf, zf, d1_xf, d1_zf, d2_xf, d2_zf

def get_spectrum_grid(betas, xf, zf, plot = False, save_plot = None):
    
    #create a container for the difference between the eigenvalues
    diffs = np.zeros( len(betas) * len(betas) ).reshape((len(betas), len(betas)))
    
    #now iterate through the grid and calculate the spectrum
    for i, beta2val in enumerate(betas):
        for j, beta1val in enumerate(betas):
            
            #evaluate the Dynamical matrix at the points
            evaled_x = xf(beta1val, beta2val)
            evaled_z = zf(beta1val, beta2val)
            
            #calculate the difference between the eigenvalues in the parameter space
            diff = 2 * np.sqrt(evaled_x**2 + evaled_z**2)
    
            #add the value to the container
            diffs[i,j] = diff
        
    #plot the grid if the plot is True
    if plot == True:
        fig = plt.figure( figsize = (6,6))
        im = plt.imshow(diffs[::-1, :], extent = [betas[0], betas[-1], betas[0], betas[-1]])
        plt.xlabel(r"$\beta_1$", fontsize = 14)
        plt.ylabel(r"$\beta_2$", fontsize = 14)
        plt.title(r"$2 \sqrt{x\left( \beta_1, \beta_2 \right)^2+z\left( \beta_1, \beta_2 \right)^2}$", fontsize = 12)
        plt.colorbar(im,fraction=0.046, pad=0.04)
        if save_plot != None:
            plt.savefig(save_plot, dpi = 500, bbox_inches='tight')
        else: 
            pass
        plt.show()
    else:
        pass
    
    #return the grid
    return diffs

def gtensor(d1_xf, d1_zf, d2_xf, d2_zf, b1_val, b2_val):

    g1_x = d1_xf(b1_val, b2_val)
    g1_z = d1_zf(b1_val, b2_val)
    g2_x = d2_xf(b1_val, b2_val)
    g2_z = d2_zf(b1_val, b2_val)
    
    g_mat = np.array([[g1_x, g2_x],
                      [g1_z, g2_z]], dtype = "float64")
    
    return g_mat

def itfunc(xf, zf, d1_xf, d1_zf, d2_xf, d2_zf, b1_val, b2_val):
    
    #evaluate everything at this specific beta1 beta2 point characterised by b1_val and b2_val
    dx = xf(b1_val, b2_val)
    dz = zf(b1_val, b2_val)
    
    #construct the d vector
    dvec = np.array([[dx],[dz]], dtype = "float64")
    
    #get the gtensor
    g_mat = gtensor(d1_xf = d1_xf, d1_zf = d1_zf, d2_xf = d2_xf, d2_zf = d2_zf, b1_val = b1_val, b2_val = b2_val)

    #get relative position of the approximate degeneracy 
    delta_betas = - (np.linalg.inv(g_mat)) @ dvec
    
    #return the delta_betas vector
    return delta_betas

def weyl_search(xf, zf, d1_xf, d1_zf, d2_xf, d2_zf, beta0, max_it = 30, prec_it = 1e-12):
    
    #set the iteration number to one
    itnum = 0
    
    #expand the the initial beta parameters
    b1_val, b2_val = beta0
    
    #calculate the first delta_betas vector
    delta_betas = itfunc(xf = xf, zf = zf, d1_xf = d1_xf, d1_zf = d1_zf, d2_xf = d2_xf,
                         d2_zf = d2_zf, b1_val = b1_val, b2_val = b2_val)
    
    #repeat this procedure if the norm of deltamass is larger than prec_it
    while np.linalg.norm(delta_betas) > prec_it and itnum < max_it : 
        
        #update the approximate location of the degeneracy
        b1_val += delta_betas[0,0]
        b2_val += delta_betas[1,0]
        
        #increase the iteration number
        itnum += 1
        
        #calculate the next delta_betas vector
        delta_betas = itfunc(xf = xf, zf = zf, d1_xf = d1_xf, d1_zf = d1_zf, d2_xf = d2_xf,
                             d2_zf = d2_zf, b1_val = b1_val, b2_val = b2_val)
        
    #return the location of the minimum and also the iteration number which can be used for debugging
    return [b1_val, b2_val], itnum

def isin(bvals, locs, loc_threshhold = 1e-10):
    #check if the new weyl point is already among the others
    
    #define a grid so that the periodicity can be get rid of
    n = (np.arange(5) - 2) * 2 * np.pi
    N1, N2 = np.meshgrid(n,n)
    Ns = np.zeros(5 * 5 * 2).reshape( (5, 5, 2) )
    Ns[:, :, 0] = N1[:, :]
    Ns[:, :, 1] = N2[:, :]
        
    #project the angles into the -pi,pi interval
    #with this we still have a periodicity which we need to take care of
    bvals[0] = np.arctan2(np.sin(bvals[0]), np.cos(bvals[0]))
    bvals[1] = np.arctan2(np.sin(bvals[1]), np.cos(bvals[1]))
        
    #if there are no weyl points found then need to add this new point
    if len(locs) == 0:
        locs.append(bvals)
        
    #else need to check whether this new point is close to another point already in the list
    else:
        
        #define new variable which indicates whether the Weyl-point is already among the others
        isin = False
        
        #iterate through all the Weyl-point already found
        for j in range(len(locs)):
            
            #take the coordinates of each point
            l1, l2 = locs[j]
            
            #calculate the of this point from the other point
            if np.any( np.sqrt( (l1 - bvals[0] - Ns[:,:,0])**2 + (l2 - bvals[1] - Ns[:,:,1])**2) < loc_threshhold):
                
                #if it is in then change isin to True and break the loop
                isin = True
                
                #break the for cycle because we already know that the Weyl-point is already among the others
                break
                    
        #if isin is true then the point is already in
        if isin: 
            
            pass
        
        #otherwise if isin False then a new Weyl-point is found and need to add it to the container
        else:
            locs.append(bvals)
            
    #return the updated locs 
    return locs

def all_point_finder(xf, zf, d1_xf, d1_zf, d2_xf, d2_zf, max_it = 30, prec_it = 1e-12, loc_threshhold = 1e-10,
                     point_itnum = 500, deg_treshhold = 1e-10):
    
    #create container for the weyl points
    locs = []
    
    #print("initially locs is: ", locs)
    
    #try to find a weyl point for each point_itnum 
    #for each itnum
    for i in range(point_itnum):
        #generate a random point in the configuration space in the [-pi, pi] interval
        b1_val, b2_val = np.random.rand(2) * 2 * np.pi - np.pi 
        
        #put the initial values into an array
        beta0 = [b1_val, b2_val]
        
        #find the local minimum closest to this initial point
        bvals, itnum = weyl_search(xf = xf, zf = zf, d1_xf = d1_xf, d1_zf = d1_zf, d2_xf = d2_xf, d2_zf = d2_zf,
                                   beta0 = beta0, max_it = max_it, prec_it = prec_it)
        #print("iteration index:", i)
        #print("bvals:", bvals)
        
        #calculate the gap in the spectrum for this local minima
        deg = 2 * np.sqrt( float( xf(bvals[0], bvals[1]) )**2 + float( zf(bvals[0], bvals[1]) )**2 )
        
        #print("degeneracy:", deg)
        
        #check whether this gap is smaller than the threshhold value
        if deg < deg_treshhold :
            
            #update the locations 
            locs = isin(bvals = bvals, locs = locs, loc_threshhold = loc_threshhold)
        
        else:
            pass
        
        #print("locations:", locs)
        
    #return the location of the weyl points
    return locs

def weyl_point_num_counter(number_of_random_systems = 100, n = 2):
    
    weylnum = np.zeros(100)
    
    for i in range(number_of_random_systems):
        start = timeit.timeit()

        #generate a new system
        xf, zf, d1_xf, d1_zf, d2_xf, d2_zf = symbolic_matrices(n = n)
        
        #get the location of the weyl points
        locs = all_point_finder(xf = xf, zf = zf, d1_xf = d1_xf, d1_zf = d1_zf, d2_xf = d2_xf, 
                                d2_zf = d2_zf)
        
        #increase the corresponding entry in the array
        weylnum[len(locs)] += 1
        
        end = timeit.timeit()
        print(end - start)
    
    return weylnum
    
def weyl_charge_calculator(d1_xf, d1_zf, d2_xf, d2_zf, locs):
    #calculate the charges corresponding to the weyl points we have found using our usual search
    charge_list = np.zeros(len(locs))
    
    #iterate through all the locations
    for idx, loc in enumerate(locs):
        
        #get the g tensor evaluated at the particular point
        gmat = gtensor(d1_xf = d1_xf, d1_zf = d1_zf, d2_xf = d2_xf, d2_zf = d2_zf, b1_val = loc[0], b2_val = loc[1])
        
        #the carge is the sign of the determinant of the g tensor
        q = np.sign(np.linalg.det(gmat))
        
        #add the winding to the container
        charge_list[idx] = q
        
    #return the charge list
    return charge_list

def obtain_42_wp_configs(number_of_random_systems = 500, n = 2, filename = "test"):

    #define containers for the 4 and 2 weyl point configuration scenarios
    wps4 = []
    wps2 = []
    
    #define container for the number statistics and fill the container with zeros till a point
    numstat = {}
    for i in range(20):
        numstat[i] = 0
        
    #for each iteration 
    for i in range(number_of_random_systems):

        #generate a new system
        x, z, xf, zf, d1_xf, d1_zf, d2_xf, d2_zf = symbolic_matrices(n = n, core = False)
        
        #get the location of the weyl points
        locs = all_point_finder(xf = xf, zf = zf, d1_xf = d1_xf, d1_zf = d1_zf, d2_xf = d2_xf, 
                                d2_zf = d2_zf)
        
        #get the length of the locs
        ln = len(locs)
        
        #also calculate the charges
        charge_list = weyl_charge_calculator(d1_xf = d1_xf, d1_zf = d1_zf, d2_xf = d2_xf, d2_zf = d2_zf, locs = locs)
        
        #keep the configuration only if the total charge of weyl points is zero
        if np.sum(charge_list) == 0:
            
            #if the number of weyl points is 4 or 2 then save the x and z components of the dynamical matrix!
            #we have 4 weyl points only when the number of minimums is  4 
            if ln == 4 and np.all(np.abs(charge_list) == 1):
                #if the nmber of weyl points is four then add the corresponding coordinatization to the container
                wps4.append([sp.srepr(x), sp.srepr(z)])
                
            #if the number of weyl points is two then add that to the coordinatization
            elif ln == 2 and np.all(np.abs(charge_list) == 1):
                wps2.append([sp.srepr(x), sp.srepr(z)])
            
            else:
                pass
            
            #add the number of weyl points to the statistics
            if ln in numstat.keys():
                numstat[ln] += 1
            else:
                numstat[ln] = 1
        else:
            pass

    #now we need to write the datas into a file
    with open(filename, "w") as file:
        file.write("System with 4 or 2 number of Weyl points in the configuration space" + " \n")
        file.write("n: " + str(n) + "\n")
        file.write("x" + "\t" + "z" + "\n" + "\n")
        file.write("4 Weyl points: " + "\n")
        for l in range(len(wps4)):
            file.write(wps4[l][0] + "\t" + wps4[l][1] + "\n")
            
        file.write("\n")
        file.write("2 Weyl points: " + "\n")
        for l in range(len(wps2)):
            file.write(wps2[l][0] + "\t" + wps2[l][1] + "\n")
            
    #and also return the statistics
    return numstat