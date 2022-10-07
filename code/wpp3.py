import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def symbolic_matrices(n, core = True):
    """
    Create the random matrix which is characterised by the Pauli X and Z components 

    Parameters
    ----------
    n : integer, the number of higher harmonics included in the random matrix
    
    core : boolean, if True then only the essential objects are returned The default is True.

    Returns
    -------
    TYPE based on 'core' the function returns a number of functions (and sympy expressions)

    """

    #define the angle parameters
    alpha1 = sp.Symbol('alpha_1', Real = True)
    alpha2 = sp.Symbol('alpha_2', Real = True)
    
    #define the angle combinations
    gammas = [alpha1, alpha2, alpha1 + alpha2, alpha2 - alpha1]
    
    #define the functional forms of x and z
    xtilde = 0
    ztilde = 0
    
    #iterate through the harmonics present in the system
    for i in range(4):
        for k in range(1, n+1):
        #add the corresponding terms to the x and z components
            xtilde += (2 * np.random.rand() - 1) * sp.sin( k * gammas[i] ) / 2**(k-1) + (2 * np.random.rand() - 1) * sp.cos( k * gammas[i] ) / 2**(k-1)
            ztilde += (2 * np.random.rand() - 1) * sp.sin( k * gammas[i] ) / 2**(k-1) + (2 * np.random.rand() - 1) * sp.cos( k * gammas[i] ) / 2**(k-1)
        
        for j in range(4):
            if i == j:
                pass
            else:
                for k in range(1, n+1):
                    for l in range(1, n+1):
                        xtilde += (2 * np.random.rand() - 1) * sp.sin(k * gammas[i]) * sp.cos(l * gammas[j]) / 8 + (2 * np.random.rand() - 1) * sp.cos(k * gammas[i]) * sp.sin(l * gammas[j]) / 8
                        ztilde += (2 * np.random.rand() - 1) * sp.sin(k * gammas[i]) * sp.cos(l * gammas[j]) / 8 + (2 * np.random.rand() - 1) * sp.cos(k * gammas[i]) * sp.sin(l * gammas[j]) / 8

    x = (xtilde - xtilde.subs({alpha1 : alpha2, alpha2 : alpha1}, simultaneous=True)) / 2
    z = (ztilde + ztilde.subs({alpha1 : alpha2, alpha2 : alpha1}, simultaneous=True)) / 2
    
    #create the symbolic derivatives of the components
    d1_x = sp.diff(x, alpha1)
    d1_z = sp.diff(z, alpha1)
    d2_x = sp.diff(x, alpha2)
    d2_z = sp.diff(z, alpha2)
    
    #make lamba functions out of these which we return and work with later on
    xf = sp.utilities.lambdify(['alpha_1', 'alpha_2'], x)
    zf = sp.utilities.lambdify(['alpha_1', 'alpha_2'], z)
    d1_xf = sp.utilities.lambdify(['alpha_1', 'alpha_2'], d1_x)
    d1_zf = sp.utilities.lambdify(['alpha_1', 'alpha_2'], d1_z)
    d2_xf = sp.utilities.lambdify(['alpha_1', 'alpha_2'], d2_x)
    d2_zf = sp.utilities.lambdify(['alpha_1', 'alpha_2'], d2_z)

    #return a bunch of objects depending on the value of core
    if core:
        #and return the lambdifyied expressions and also the symbols
        return xf, zf, d1_xf, d1_zf, d2_xf, d2_zf
    
    else:
        #if core is False then also return the sympy expressions x and z
        return x, z, xf, zf, d1_xf, d1_zf, d2_xf, d2_zf

def get_spectrum_grid(alphas, xf, zf, plot = False, save_plot = None):
    """
    Create a discrete grid on which the splitting is evaluated. 

    Parameters
    ----------
    betas : numpy array, the finite grid along the angle axis. This array is used to create the full 2D grid.
    
    xf : lambda function, if evaluated at a certain (beta1, beta2) point, gives the Pauli X component of the random matrix.
    
    zf : lambda function, if evaluated at a certain (beta1, beta2) point, gives the Pauli Z component of the random matrix.

    plot : boolean, if True then a plot is created from the splitting values. The default is False.
    
    save_plot : string, if not None, then the plot gets saved. The default is None.

    Returns
    -------
    diffs : 2D numpy array, contains the splitting values.

    """
    
    #create a container for the splitting
    diffs = np.zeros( len(alphas) * len(alphas) ).reshape((len(alphas), len(alphas)))
    
    #now iterate through the grid and calculate the spectrum
    for i, alpha2val in enumerate(alphas):
        for j, alpha1val in enumerate(alphas):
            
            #evaluate the Dynamical matrix at the points
            evaled_x = xf(alpha1val, alpha2val)
            evaled_z = zf(alpha1val, alpha2val)
            
            #calculate the splitting
            diff = 2 * np.sqrt(evaled_x**2 + evaled_z**2)
    
            #add the value to the container
            diffs[i,j] = diff
        
    #plot the grid if the plot is True
    if plot == True:
        #create figure
        fig = plt.figure( figsize = (6,6))
        
        #plot the values (note the ::-1 !)
        im = plt.imshow(diffs[::-1, :], extent = [alphas[0], alphas[-1], alphas[0], alphas[-1]])
        
        #add labels 
        plt.xlabel(r"$\alpha_1$", fontsize = 14)
        plt.ylabel(r"$\alpha_2$", fontsize = 14)
        
        #add title
        plt.title(r"$2 \sqrt{x\left( \alpha_1, \alpha_2 \right)^2+z\left( \alpha_1, \alpha_2 \right)^2}$", fontsize = 12)
        
        #rescale the colorbar
        plt.colorbar(im,fraction=0.046, pad=0.04)
        
        #save plot if needed
        if save_plot != None:
            plt.savefig(save_plot, dpi = 500, bbox_inches='tight')
        else: 
            pass
        
        #show the figure anyways
        plt.show()
    else:
        pass
    
    #return the grid
    return diffs

def gtensor(d1_xf, d1_zf, d2_xf, d2_zf, alpha1val, alpha2val):
    """
    Calculate the g-tensor of the random matrix

    Parameters
    ----------
    d1_xf : lambda function, the derivative of the Pauli X component of the random matrix w.r.t. beta1.
    
    d1_zf : lambda function, the derivative of the Pauli Z component of the random matrix w.r.t. beta1.
    
    d2_xf : lambda function, the derivative of the Pauli X component of the random matrix w.r.t. beta2.
    
    d2_zf : lambda function, the derivative of the Pauli Z component of the random matrix w.r.t. beta2.
    
    b1_val : float, the beta1 point at which the g-tensor is to be obtained.
    
    b2_val : float, the beta2 point at which the g-tensor is to be obtained.

    Returns
    -------
    g_mat : 2D numpy array, the g-tensor corresponding to the point in the configuration space

    """

    #evaluate the components of the g-tensor
    g1_x = d1_xf(alpha1val, alpha2val)
    g1_z = d1_zf(alpha1val, alpha2val)
    g2_x = d2_xf(alpha1val, alpha2val)
    g2_z = d2_zf(alpha1val, alpha2val)
    
    #construct the g-tensor
    g_mat = np.array([[g1_x, g2_x],
                      [g1_z, g2_z]], dtype = "float64")
    
    return g_mat

def itfunc(xf, zf, d1_xf, d1_zf, d2_xf, d2_zf, alpha1val, alpha2val):
    """
    Iteration function which is used during the Weyl-point search.

    Parameters
    ----------
    xf : lambda function, if evaluated at a certain (beta1, beta2) point, gives the Pauli X component of the random matrix.
    
    zf : lambda function, if evaluated at a certain (beta1, beta2) point, gives the Pauli Z component of the random matrix.
    
    d1_xf : lambda function, the derivative of the Pauli X component of the random matrix w.r.t. beta1.
    
    d1_zf : lambda function, the derivative of the Pauli Z component of the random matrix w.r.t. beta1.
    
    d2_xf : lambda function, the derivative of the Pauli X component of the random matrix w.r.t. beta2.
    
    d2_zf : lambda function, the derivative of the Pauli Z component of the random matrix w.r.t. beta2.
    
    b1_val : float, the beta1 point at which the g-tensor is to be obtained.
    
    b2_val : float, the beta2 point at which the g-tensor is to be obtained.

    Returns
    -------
    delta_betas : numpy array, the change in the beta values to get closer to the zero splitting

    """
    
    #evaluate everything at this specific point characterised by b1_val and b2_val
    dx = xf(alpha1val, alpha2val)
    dz = zf(alpha1val, alpha2val)
    
    #construct the d vector
    dvec = np.array([[dx],[dz]], dtype = "float64")
    
    #get the gtensor
    g_mat = gtensor(d1_xf = d1_xf, d1_zf = d1_zf, d2_xf = d2_xf, d2_zf = d2_zf, alpha1val = alpha1val, alpha2val = alpha2val)

    #get relative position of the approximate degeneracy 
    delta_alphas = - (np.linalg.inv(g_mat)) @ dvec
    
    #return the delta_betas vector
    return delta_alphas

def weyl_search(xf, zf, d1_xf, d1_zf, d2_xf, d2_zf, alpha0, max_it = 30, prec_it = 1e-12):
    """
    Function that carries out the Weyl-point search starting from a random point specified by beta0

    Parameters
    ----------
    xf : lambda function, if evaluated at a certain (beta1, beta2) point, gives the Pauli X component of the random matrix.
    
    zf : lambda function, if evaluated at a certain (beta1, beta2) point, gives the Pauli Z component of the random matrix.
    
    d1_xf : lambda function, the derivative of the Pauli X component of the random matrix w.r.t. beta1.
    
    d1_zf : lambda function, the derivative of the Pauli Z component of the random matrix w.r.t. beta1.
    
    d2_xf : lambda function, the derivative of the Pauli X component of the random matrix w.r.t. beta2.
    
    d2_zf : lambda function, the derivative of the Pauli Z component of the random matrix w.r.t. beta2.
    
    beta0 : numpy array, the (beta1,beta2) point from which the random search starts
    
    max_it : integer, the maximal number of iterations in the search. The default is 30.
    
    prec_it : float, characterises the termination condition of the search. The default is 1e-12.

    Returns
    -------
    list : list of floats, contains the approximate location of the Weyl-point that was obtained with the search.
    
    itnum : integer, the number iterations done in the search.

    """
    
    #set the iteration number to zero
    itnum = 0
    
    #expand the the initial beta parameters
    alpha1val, alpha2val = alpha0
    
    #calculate the first delta_betas vector
    delta_alphas = itfunc(xf = xf, zf = zf, d1_xf = d1_xf, d1_zf = d1_zf, d2_xf = d2_xf,
                         d2_zf = d2_zf, alpha1val = alpha1val, alpha2val = alpha2val)
    
    #repeat this procedure if the norm of deltamass is larger than prec_it and if the number of iterations is not larger than the maximal allowed value
    while np.linalg.norm(delta_alphas) > prec_it and itnum < max_it : 
        
        #update the approximate location of the degeneracy
        alpha1val += delta_alphas[0,0]
        alpha2val += delta_alphas[1,0]
        
        #increase the iteration number
        itnum += 1
        
        #calculate the next delta_betas vector
        delta_alphas = itfunc(xf = xf, zf = zf, d1_xf = d1_xf, d1_zf = d1_zf, d2_xf = d2_xf,
                             d2_zf = d2_zf, alpha1val = alpha1val, alpha2val = alpha2val)
        
    #return the location of the minimum and also the iteration number which can be used for debugging
    return [alpha1val, alpha2val], itnum

def isin(alphavals, locs, loc_threshhold = 1e-10):
    """
    Function that checks whether a Weyl-point location is already present among the ones the search already found.

    Parameters
    ----------
    bvals : list of floats, contains the approximate location of the Weyl-point that was obtained with the search.
    
    locs :  list of list of floats, each element is a list that contains the location of a Weyl-point. 
    
    loc_threshhold : float, characterises the minimal distance between two Weyl-points. The default is 1e-10.

    Returns
    -------
    locs : list of list of floats, the updated Weyl-point locations.

    """
    
    #define a grid so that the periodicity can be get rid of
    #this part is kind of hard to describe, I think np.arange(3)-1 would be equally good
    n = (np.arange(5) - 2) * 2 * np.pi
    N1, N2 = np.meshgrid(n,n)
    Ns = np.zeros(5 * 5 * 2).reshape( (5, 5, 2) )
    Ns[:, :, 0] = N1[:, :]
    Ns[:, :, 1] = N2[:, :]
        
    #project the angles into the -pi,pi interval
    #with this we still have a periodicity which we need to take care of
    alphavals[0] = np.arctan2(np.sin(alphavals[0]), np.cos(alphavals[0]))
    alphavals[1] = np.arctan2(np.sin(alphavals[1]), np.cos(alphavals[1]))
    
    #if there are no weyl points found then need to add this new point
    if len(locs) == 0:
        locs.append(alphavals)
        
    #else need to check whether this new point is close to another point already in the list
    else:
        
        #define new variable which indicates whether the Weyl-point is already among the others
        isin = False
        
        #iterate through all the Weyl-point already found
        for j in range(len(locs)):
            
            #take the coordinates of each point
            l1, l2 = locs[j]
            
            #calculate the of this point from the other point
            if np.any( np.sqrt( (l1 - alphavals[0] - Ns[:,:,0])**2 + (l2 - alphavals[1] - Ns[:,:,1])**2) < loc_threshhold):
                
                #if it is in then change isin to True and break the loop
                isin = True
                
                #break the for cycle because we already know that the Weyl-point is already among the others
                break
                    
        #if isin is true then the point is already in
        if isin: 
            
            pass
        
        #otherwise if isin False then a new Weyl-point is found and need to add it to the container
        else:
            locs.append(alphavals)
            
    #return the updated locs 
    return locs

def all_point_finder(xf, zf, d1_xf, d1_zf, d2_xf, d2_zf, max_it = 30, prec_it = 1e-12, loc_threshhold = 1e-10,
                     point_itnum = 500, deg_treshhold = 1e-10):
    """
    Function that finds all the Weyl-points in the configuration space for a given random matrix.

    Parameters
    ----------
    xf : lambda function, if evaluated at a certain (beta1, beta2) point, gives the Pauli X component of the random matrix.
    
    zf : lambda function, if evaluated at a certain (beta1, beta2) point, gives the Pauli Z component of the random matrix.
    
    d1_xf : lambda function, the derivative of the Pauli X component of the random matrix w.r.t. beta1.
    
    d1_zf : lambda function, the derivative of the Pauli Z component of the random matrix w.r.t. beta1.
    
    d2_xf : lambda function, the derivative of the Pauli X component of the random matrix w.r.t. beta2.
    
    d2_zf : lambda function, the derivative of the Pauli Z component of the random matrix w.r.t. beta2.
    
    max_it : integer, the maximal number of iterations in the search. The default is 30.
    
    prec_it : float, characterises the termination condition of the search. The default is 1e-12.
    
    loc_threshhold : float, characterises the minimal distance between two Weyl-points. The default is 1e-10.
    
    point_itnum : integer, the number of random locations from which the Weyl-point search is started. The default is 500.
    
    deg_treshhold : float, specifies the maximal splitting accepted for a Weyl-point. The default is 1e-10.

    Returns
    -------
    locs : list of list of floats, the updated Weyl-point locations.

    """
    
    #create container for the weyl points
    locs = []
    
    #print("initially locs is: ", locs)
    
    #start to generate random starting points for the Weyl-point search
    for i in range(point_itnum):
        
        #generate a random point in the configuration space in the [-pi, pi] interval
        alpha1val, alpha2val = np.random.rand(2) * 2 * np.pi - np.pi 
        
        #put the initial values into an array
        alpha0 = [alpha1val, alpha2val]
        
        #find the local minimum closest to this initial point
        alphavals, itnum = weyl_search(xf = xf, zf = zf, d1_xf = d1_xf, d1_zf = d1_zf, d2_xf = d2_xf, d2_zf = d2_zf,
                                   alpha0 = alpha0, max_it = max_it, prec_it = prec_it)
        
        #print("iteration index:", i)
        #print("bvals:", bvals)
        
        #calculate the gap in the spectrum for this local minima
        deg = 2 * np.sqrt( float( xf(alphavals[0], alphavals[1]) )**2 + float( zf(alphavals[0], alphavals[1]) )**2 )
        
        #print("degeneracy:", deg)
        
        #check whether this gap is smaller than the threshhold value
        if deg < deg_treshhold :
            
            #update the locations 
            locs = isin(alphavals = alphavals, locs = locs, loc_threshhold = loc_threshhold)
        
        else:
            pass
        
        #print("locations:", locs)
        
    #return the location of the weyl points
    return locs
    
def weyl_charge_calculator(d1_xf, d1_zf, d2_xf, d2_zf, locs):
    """
    Function that calculates the charge of the Weyl-points found during the search.

    Parameters
    ----------
    d1_xf : lambda function, the derivative of the Pauli X component of the random matrix w.r.t. beta1.
    
    d1_zf : lambda function, the derivative of the Pauli Z component of the random matrix w.r.t. beta1.
    
    d2_xf : lambda function, the derivative of the Pauli X component of the random matrix w.r.t. beta2.
    
    d2_zf : lambda function, the derivative of the Pauli Z component of the random matrix w.r.t. beta2.
    
    locs : list of list of floats, the updated Weyl-point locations.
    
    Returns
    -------
    charge_list : list of integers, the charges corresponding to the Weyl-points.

    """
    #calculate the charges corresponding to the weyl points we have found using our usual search
    charge_list = np.zeros(len(locs))
    
    #iterate through all the locations
    for idx, loc in enumerate(locs):
        
        #get the g tensor evaluated at the particular point
        gmat = gtensor(d1_xf = d1_xf, d1_zf = d1_zf, d2_xf = d2_xf, d2_zf = d2_zf, alpha1val = loc[0], alpha2val = loc[1])
        
        #the carge is the sign of the determinant of the g tensor
        q = np.sign(np.linalg.det(gmat))
        
        #add the winding to the container
        charge_list[idx] = q
        
    #return the charge list
    return charge_list

def obtain_42_wp_configs(number_of_random_systems = 500, n = 2, filename = "test"):
    """
    Function that finds random systems with 4 and 2 Weyl-points in the configuration space.

    Parameters
    ----------
    number_of_random_systems : integer, the number of random systems to generate. The default is 500.
    
    n : integer, the number of higher harmonics included in the random matrix. The default is 2.
    
    filename : string, the name of the file into which the data should be saved. The default is "test".

    Returns
    -------
    numstat : dictionary, contains statistics about the number of occurences of a given Weyl-point number.

    """

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