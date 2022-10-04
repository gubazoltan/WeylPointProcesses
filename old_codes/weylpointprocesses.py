import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import timeit

def symbolic_matrices(n):
    """
    Create the system with harmonics up to n. Instead of the matrices only the relevant Pauli X and Z components are generated. 
    

    Parameters
    ----------
    n : integer, the order of higher harmonics present in the decomposition of the matrix elements. 

    Returns
    -------
    x : sympy expression, the symoblic expression which characterises the Pauli X component of the dynamical matrix.
    
    z : sympy expression, the symoblic expression which characterises the Pauli Z component of the dynamical matrix.

    d1_x : sympy expression, the symbolic expression which characterises the derivative of the Pauli X component of the dynamical matrix
        with respect to the first angle variable. 
    
    d1_z : sympy expression, the symbolic expression which characterises the derivative of the Pauli Z component of the dynamical matrix
        with respect to the first angle variable. 
        
    d2_x : sympy expression, the symbolic expression which characterises the derivative of the Pauli X component of the dynamical matrix
        with respect to the second angle variable. 
        
    d2_z : sympy expression, the symbolic expression which characterises the derivative of the Pauli Z component of the dynamical matrix
        with respect to the second angle variable. 
        
    beta1 : sympy symbol, the first angle variable. This variable is totally symmetric under the angle reflection of the system
    
    beta2 : sympy symbol, the second angle variable. This variable is totally antisymmetric under the angle reflection of the system

    """

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
    
    #return the matrices and also the symbols
    return x, z, d1_x, d1_z, d2_x, d2_z, beta1, beta2

def get_spectrum_grid(betas, x, z, beta1, beta2, plot = False):
    """
    Create a grid containing the difference of the eigenvalues of a system characterised by the Pauli X and Z components x and z.

    Parameters
    ----------
    betas : numpy array, the finite grid along the angle axis. This array is used to create the full 2D grid.
    
    x : sympy expression, the symoblic expression which characterises the Pauli X component of the dynamical matrix.
    
    z : sympy expression, the symoblic expression which characterises the Pauli Z component of the dynamical matrix.

    beta1 : sympy symbol, the first angle variable. This variable is totally symmetric under the angle reflection of the system
    
    beta2 : sympy symbol, the second angle variable. This variable is totally antisymmetric under the angle reflection of the system

    plot : boolean, if True, then also plot the grid, if False then don't plot the grid.

    Returns
    -------
    diffs : 2D numpy array of floats, contains the difference between the eigenvalues of the system.

    """
    
    #create a container for the difference between the eigenvalues
    diffs = np.zeros( len(betas) * len(betas) ).reshape((len(betas), len(betas)))
    
    #create lambdafunction out of the symbolic components (evaluation is faster this way)
    x_func = sp.utilities.lambdify(['beta_1', 'beta_2'], x)
    z_func = sp.utilities.lambdify(['beta_1', 'beta_2'], z)
    
    #now iterate through the grid and calculate the spectrum
    for i, beta2val in enumerate(betas):
        for j, beta1val in enumerate(betas):
            
            #evaluate the Dynamical matrix at the points
            evaled_x = x_func(beta1val, beta2val)
            evaled_z = z_func(beta1val, beta2val)
            
            #calculate the difference between the eigenvalues in the parameter space
            diff = 2 * np.sqrt(evaled_x**2 + evaled_z**2)
    
            #add the value to the container
            diffs[i,j] = diff
        
    #plot the grid if the plot is True
    if plot == True:
        fig = plt.figure()
        plt.imshow(diffs[::-1, :], extent = [betas[0], betas[-1], betas[0], betas[-1]])
        plt.xlabel(r"$\beta_1$", fontsize = 12)
        plt.ylabel(r"$\beta_2$", fontsize = 12)

    else:
        pass
    
    #return the grid
    return diffs

def itfunc(x, z, d1_x, d1_z, d2_x, d2_z, beta1, beta2, b1_val, b2_val):
    """
    Function which we need to iterate in the Weyl-point serach method.

    Parameters
    ----------
    x : sympy expression, the symoblic expression which characterises the Pauli X component of the dynamical matrix.
    
    z : sympy expression, the symoblic expression which characterises the Pauli Z component of the dynamical matrix.

    d1_x : sympy expression, the symbolic expression which characterises the derivative of the Pauli X component of the dynamical matrix
        with respect to the first angle variable. 
    
    d1_z : sympy expression, the symbolic expression which characterises the derivative of the Pauli Z component of the dynamical matrix
        with respect to the first angle variable. 
        
    d2_x : sympy expression, the symbolic expression which characterises the derivative of the Pauli X component of the dynamical matrix
        with respect to the second angle variable. 
        
    d2_z : sympy expression, the symbolic expression which characterises the derivative of the Pauli Z component of the dynamical matrix
        with respect to the second angle variable. 
        
    beta1 : sympy symbol, the first angle variable. This variable is totally symmetric under the angle reflection of the system
    
    beta2 : sympy symbol, the second angle variable. This variable is totally antisymmetric under the angle reflection of the system
    
    b1_val : float, the value of the first angle variable at which the Weyl-point search is currently at.
    
    b2_val : float, the value of the second angle variable at which the Weyl-point search is currently at.

    Returns
    -------
    delta_betas : numpy array of floats, the array contains the change in the angle variables after the iteration method

    """
    
    #evaluate everything at this specific beta1 beta2 point characterised by b1_val and b2_val
    dx = x.evalf(subs = {beta1: b1_val, beta2 : b2_val})
    dz = z.evalf(subs = {beta1: b1_val, beta2 : b2_val})
    
    g1_x = d1_x.evalf(subs = {beta1: b1_val, beta2 : b2_val})
    g1_z = d1_z.evalf(subs = {beta1: b1_val, beta2 : b2_val})
    g2_x = d2_x.evalf(subs = {beta1: b1_val, beta2 : b2_val})
    g2_z = d2_z.evalf(subs = {beta1: b1_val, beta2 : b2_val})
    
    #construct the d vector and the g tensor
    #here it is important to have the data type as float64 otherwise the g matrix will be a sympy object for which this inversion
    #cannot be used
    dvec = np.array([[dx],[dz]], dtype = "float64")
    g_mat = np.array([[g1_x, g2_x],
                      [g1_z, g2_z]], dtype = "float64")

    #get relative position of the approximate degeneracy 
    delta_betas = - (np.linalg.inv(g_mat)) @ dvec
    
    #return the delta_betas vector
    return delta_betas

def weyl_search(x, z, d1_x, d1_z, d2_x, d2_z, beta1, beta2, max_it, prec_it, beta0):
    """
    The Weyl-point search method invented by Gy. Frank. 

    Parameters
    ----------
    x : sympy expression, the symoblic expression which characterises the Pauli X component of the dynamical matrix.
    
    z : sympy expression, the symoblic expression which characterises the Pauli Z component of the dynamical matrix.

    d1_x : sympy expression, the symbolic expression which characterises the derivative of the Pauli X component of the dynamical matrix
        with respect to the first angle variable. 
    
    d1_z : sympy expression, the symbolic expression which characterises the derivative of the Pauli Z component of the dynamical matrix
        with respect to the first angle variable. 
        
    d2_x : sympy expression, the symbolic expression which characterises the derivative of the Pauli X component of the dynamical matrix
        with respect to the second angle variable. 
        
    d2_z : sympy expression, the symbolic expression which characterises the derivative of the Pauli Z component of the dynamical matrix
        with respect to the second angle variable. 
        
    beta1 : sympy symbol, the first angle variable. This variable is totally symmetric under the angle reflection of the system
    
    beta2 : sympy symbol, the second angle variable. This variable is totally antisymmetric under the angle reflection of the system
    
    max_it : integer, the maximum number of iteration that the search does before terminating
    
    prec_it : float, the search stops whenever the 2-norm of the change of the beta variables becomes smaller than prec_it.
    
    beta0 : array of floats, the starting point of the iteration.

    Returns
    -------
    list: list of floats, the location of the Weyl-points as obtained by the search
    
    itnum : integer, the number of iterations needed to find the Weyl-point. Can be used for debugging.

    """
    
    #set the iteration number to one
    itnum = 0
    
    #expand the the initial beta parameters
    b1_val, b2_val = beta0
    
    #calculate the first delta_betas vector
    delta_betas = itfunc(x = x, z = z, d1_x = d1_x, d1_z = d1_z, d2_x = d2_x,
                         d2_z = d2_z, beta1 = beta1, beta2 = beta2, b1_val = b1_val, b2_val = b2_val)
    
    #repeat this procedure if the norm of deltamass is larger than prec_it
    while np.linalg.norm(delta_betas) > prec_it and itnum < max_it : 
        
        #update the approximate location of the degeneracy
        b1_val += delta_betas[0,0]
        b2_val += delta_betas[1,0]
        
        #increase the iteration number
        itnum += 1
        
        #calculate the next delta_betas vector
        delta_betas = itfunc(x = x, z = z, d1_x = d1_x, d1_z = d1_z, d2_x = d2_x,
                             d2_z = d2_z, beta1 = beta1, beta2 = beta2, b1_val = b1_val, b2_val = b2_val)
        
    #return the location of the minimum and also the iteration number which can be used for debugging
    return [b1_val, b2_val], itnum

def all_point_finder(x, z, d1_x, d1_z, d2_x, d2_z, beta1, beta2, max_it = 20, prec_it = 1e-8, loc_threshhold = 1e-6,
                     point_itnum = 100, deg_treshhold = 1e-10):
    """
    Finds all (hopefully) Weyl-points in the configuration space 

    Parameters
    ----------
    x : sympy expression, the symoblic expression which characterises the Pauli X component of the dynamical matrix.
    
    z : sympy expression, the symoblic expression which characterises the Pauli Z component of the dynamical matrix.

    d1_x : sympy expression, the symbolic expression which characterises the derivative of the Pauli X component of the dynamical matrix
        with respect to the first angle variable. 
    
    d1_z : sympy expression, the symbolic expression which characterises the derivative of the Pauli Z component of the dynamical matrix
        with respect to the first angle variable. 
        
    d2_x : sympy expression, the symbolic expression which characterises the derivative of the Pauli X component of the dynamical matrix
        with respect to the second angle variable. 
        
    d2_z : sympy expression, the symbolic expression which characterises the derivative of the Pauli Z component of the dynamical matrix
        with respect to the second angle variable. 
        
    beta1 : sympy symbol, the first angle variable. This variable is totally symmetric under the angle reflection of the system
    
    beta2 : sympy symbol, the second angle variable. This variable is totally antisymmetric under the angle reflection of the system
    
    max_it : integer, the maximum number of iteration that the search does before terminating. The default is 20.
    
    prec_it : float, the search stops whenever the 2-norm of the change of the beta variables becomes smaller than prec_it. The default is 1e-8.
    
    loc_threshhold : float, the minimal distance above which two Weyl-points is considered to be different. The default is 1e-6.
    
    point_itnum : integer, the number of random beta variables from which the Weyl-point search method is started. The default is 100.
    
    deg_treshhold : float, the minimum difference between the eigenvalues of the system. The spectrum is considered to be degenerate
        in the configuration space if the splitting at that point is smaller than this value. The default is 1e-10.

    Returns
    -------
    locs : list of arrays, contains the locations of all the Weyl-points. 

    """
    
    #define a grid so that the periodicity can be get rid of
    n = (np.arange(5) - 2) * 2 * np.pi
    N1, N2 = np.meshgrid(n,n)
    Ns = np.zeros(5 * 5 * 2).reshape( (5, 5, 2) )
    Ns[:, :, 0] = N1[:, :]
    Ns[:, :, 1] = N2[:, :]
    
    #create container for the weyl points
    locs = []
    
    #print("initially locs is: ", locs)
    
    #try to find a weyl point for each point_itnum 
    #for each itnum
    for i in range(point_itnum):
        #generate a random point in the configuration space in the [-pi, pi] interval
        b1_val, b2_val = np.random.rand(2)*2*np.pi - np.pi 
        
        #put the initial values into an array
        beta0 = [b1_val, b2_val]
        
        #find the local minimum closest to this initial point
        bvals, itnum = weyl_search(x = x, z = z, d1_x = d1_x, d1_z = d1_z, d2_x = d2_x, d2_z = d2_z,
                                   beta1 = beta1, beta2 = beta2, max_it = max_it, prec_it = prec_it, beta0 = beta0)
        #print("iteration index:", i)
        #print("bvals:", bvals)
        
        #project the angles into the -pi,pi interval
        #with this we still have a periodicity which we need to take care of
        bvals[0] = np.arctan2(np.sin(bvals[0]), np.cos(bvals[0]))
        bvals[1] = np.arctan2(np.sin(bvals[1]), np.cos(bvals[1]))
        
        #calculate the gap in the spectrum for this local minima
        deg = 2 * np.sqrt( float((x.evalf(subs = {beta1 : bvals[0], beta2: bvals[1]}))**2) + float((z.evalf(subs = {beta1 : bvals[0], beta2: bvals[1]}))**2))
        
        #print("degeneracy:", deg)
        
        #check whether this gap is smaller than the threshhold value
        if deg < deg_treshhold :
            
            #if there are no Weyl-points yet in the system then this point minima is added
            if len(locs) == 0:
                locs.append(bvals)
        
            #otherwise one needs to check whether this is actually a new weyl point and it is far away from the other Weyl-points 
            else:
                
                #define new variable which indicates whether the Weyl-point is already among the others
                isin = False
                
                #iterate through all the Weyl-point already found
                for j in range(len(locs)):
                    
                    #take the coordinates of each point
                    l1, l2 = locs[j]
                    
                    #calculate the of this point from the other point
                    if np.any( np.sqrt( (l1 - bvals[0]-Ns[:,:,0])**2 + (l2 - bvals[1]-Ns[:,:,1])**2) < loc_threshhold):
                        
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
        x, z, d1_x, d1_z, d2_x, d2_z, beta1, beta2 = symbolic_matrices(n = n)
        
        #get the location of the weyl points
        locs = all_point_finder(x = x, z = z, d1_x = d1_x, d1_z = d1_z, d2_x = d2_x, 
                                d2_z = d2_z, beta1 = beta1, beta2 = beta2)
        
        #increase the corresponding entry in the array
        weylnum[len(locs)] += 1
        
        end = timeit.timeit()
        print(end - start)
    return weylnum

def normalize(v):
    if np.linalg.norm(v) == 0:
        return v
    else:
        #return the normalized vector
        return v / np.linalg.norm(v)
    
def dvector(x, z, beta1, beta2, bvals):
    
    #get the components
    xval = x.evalf(subs = {beta1 : bvals[0], beta2 : bvals[1]})
    zval = z.evalf(subs = {beta1 : bvals[0], beta2 : bvals[1]})
    
    #define the d vector
    d = [float(xval), float(zval)]
    
    #normalize the d vector
    dnormed = normalize(d)
    
    #return the normalized d vector
    return dnormed

def winding(x, z, beta1, beta2, bvals0, b_radius = 1e-7, phi_num = 30):
    
    #calculate the winding of the vector field around the points defined in bvals
    
    #get the possible angle values
    phis = np.linspace(0, 2 * np.pi, phi_num, endpoint = False)
    
    #define the possible delta beta1 and delta beta2 values
    db1_vals = b_radius * np.cos(phis)
    db2_vals = b_radius * np.sin(phis)
    
    #define the container for the winding
    Q = 0
    
    #define container for the phases
    phases = np.zeros(phi_num)

    #iterate through the loop around the point 
    for idx, db1_val in enumerate(db1_vals):
        #obtain the dm2 value
        db2_val = db2_vals[idx]
        
        #update the bvals
        bvals = [bvals0[0] + db1_val, bvals0[1] + db2_val]
        
        #obtain the corresponding normalized d vector
        dvec = dvector(x = x, z = z, beta1 = beta1, beta2 = beta2, bvals = bvals)
        
        #calculate the phase of the dvector        
        #add the phase to the container
        phases[idx] = np.arctan2( dvec[1], dvec[0] )
        
    #now calculate the winding of the dvectors
    #iterate through the angles
    for i in range(phi_num):
        #get the angle between the phases and add it to the winding 
        Q += np.angle( np.exp( 1.j * ( phases[i] - phases[ i - 1] ) ) )
    
    #divide the winding container by 2pi
    Q = Q / (2 * np.pi)
    
    #return the winding
    return Q

def weyl_charge_calculator(x, z, beta1, beta2, locs):
    #calculate the charges corresponding to the weyl points we have found using our usual search
    charge_list = np.zeros(len(locs))
    
    #iterate through all the locations
    for idx, loc in enumerate(locs):
        
        #get the winding
        q = winding(x = x, z = z, beta1 = beta1, beta2 = beta2, bvals0 = loc)
        
        #round the winding to an integer for better precision 
        q = round(q)
        
        #add the winding to the container
        charge_list[idx] = q
        
    #return the charge list
    return charge_list
    
def obtain_42_wp_configs(number_of_random_systems = 100, n = 2, filename = "test"):

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
        x, z, d1_x, d1_z, d2_x, d2_z, beta1, beta2 = symbolic_matrices(n = n)
        
        #get the location of the weyl points
        locs = all_point_finder(x = x, z = z, d1_x = d1_x, d1_z = d1_z, d2_x = d2_x, 
                                d2_z = d2_z, beta1 = beta1, beta2 = beta2)
        
        #get the length of the locs
        ln = len(locs)
        
        #also calculate the charges
        charge_list = weyl_charge_calculator(x = x, z = z, beta1 = beta1, beta2 = beta2, locs = locs)
        
        #we keep the configuration only if the total charge of weyl points is zero
        #and the number of weyl poits found is the same as the locations found
        if np.sum(charge_list) == 0 and len(charge_list) == ln:
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