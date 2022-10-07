import numpy as np
import sympy as sp
import wpp3

def interpolator(x1, z1, x2, z2, alpha1, alpha2):
    """
    Function that creates symbolic expressions with which the interpolation can be done

    Parameters
    ----------
    x1 : sympy expression, if evaluated at a certain (beta1, beta2) point, gives the Pauli X component of the first random matrix.
    
    z1 : sympy expression, if evaluated at a certain (beta1, beta2) point, gives the Pauli Z component of the first random matrix.
    
    x2 : sympy expression, if evaluated at a certain (beta1, beta2) point, gives the Pauli X component of the second random matrix.
    
    z2 : sympy expression, if evaluated at a certain (beta1, beta2) point, gives the Pauli Z component of the second random matrix.
    
    beta1 : sympy symbol, the beta1 angle variable which is used in the expressions.
    
    beta2 : sympy symbol, the beta2 angle variable which is used in the expressions.

    Returns
    -------
    xt : sympy expression, if evaluated at a certain (t, beta1, beta2) point, gives the Pauli X component of the interpolating matrix at t.
    
    zt : sympy expression, if evaluated at a certain (t, beta1, beta2) point, gives the Pauli Z component of the interpolating matrix at t.
    
    d1_xt : sympy expression, derivative of the Pauli X component w.r.t beta1 at t.
    
    d1_zt : sympy expression, derivative of the Pauli Z component w.r.t beta1 at t.
    
    d2_xt : sympy expression, derivative of the Pauli X component w.r.t beta2 at t.
    
    d2_zt : sympy expression, derivative of the Pauli Z component w.r.t beta2 at t.
    
    t : sympy expression, the interpolating variable.

    """
    
    #defien a new symbol,t, which characterises the interpolation between the two systems
    t = sp.Symbol('t')
    
    #define the interpolating expressions
    xt = (1 - t) * x1 + t * x2  
    zt = (1 - t) * z1 + t * z2
    
    #find the symbolic derivatives of the components
    d1_xt = sp.diff(xt, alpha1)
    d1_zt = sp.diff(zt, alpha1)
    d2_xt = sp.diff(xt, alpha2)
    d2_zt = sp.diff(zt, alpha2)
    
    #return all the stuff we need
    return xt, zt, d1_xt, d1_zt, d2_xt, d2_zt, t

def three_point_process(x1, z1, x2, z2, alpha1, alpha2, ts, max_it = 30, prec_it = 1e-12, loc_threshhold = 1e-10,
                     point_itnum = 500, deg_treshhold = 1e-10):
    """
    Function that interpolates between two systems specified by x1, z1 and x2, z2. 

    Parameters
    ----------
    x1 : sympy expression, the Pauli X component of the first system.
    
    z1 : sympy expression, the Pauli Z component of the first system.
    
    x2 : sympy expression, the Pauli X component of the second system.
    
    z2 : sympy expression, the Pauli Z component of the second system.
    
    beta1 : sympy symbol, the beta1 angle variable which is used in the expressions.
    
    beta2 : sympy symbol, the beta2 angle variable which is used in the expressions.
    
    ts : list of floats, the possible values of the interpolating variable.
    
    max_it : integer, the maximal number of iterations in the search. The default is 30.
    
    prec_it : float, characterises the termination condition of the search. The default is 1e-12.
    
    loc_threshhold : float, characterises the minimal distance between two Weyl-points. The default is 1e-10.
    
    point_itnum : integer, the number of random locations from which the Weyl-point search is started. The default is 500.
    
    deg_treshhold : float, specifies the maximal splitting accepted for a Weyl-point. The default is 1e-10.

    Returns
    -------
    locs_t : list of locs, each entry is a list of Weyl-point locations. 
    
    charges_t : list of charges, each entry is a list of Weyl-point charges.

    """
    
    #define the interpolating system
    xt, zt, d1_xt, d1_zt, d2_xt, d2_zt, t = interpolator(x1 = x1, z1 = z1, x2 = x2, z2 = z2, alpha1 = alpha1, alpha2 = alpha2)
    
    #define container for the locations and for the charges 
    locs_t = []
    charges_t = []
    
    for idx, tval in enumerate(ts):
        
        #evaluate everything at this given tvalue at this given t value
        x = xt.evalf(subs = {t : tval})
        z = zt.evalf(subs = {t : tval})
        
        d1_x = d1_xt.evalf(subs = {t : tval})
        d1_z = d1_zt.evalf(subs = {t : tval})
        d2_x = d2_xt.evalf(subs = {t : tval})
        d2_z = d2_zt.evalf(subs = {t : tval})

        #make lamba functions out of these 
        xf = sp.utilities.lambdify(['alpha_1', 'alpha_2'], x)
        zf = sp.utilities.lambdify(['alpha_1', 'alpha_2'], z)
        d1_xf = sp.utilities.lambdify(['alpha_1', 'alpha_2'], d1_x)
        d1_zf = sp.utilities.lambdify(['alpha_1', 'alpha_2'], d1_z)
        d2_xf = sp.utilities.lambdify(['alpha_1', 'alpha_2'], d2_x)
        d2_zf = sp.utilities.lambdify(['alpha_1', 'alpha_2'], d2_z)
            
        #get the location of the weyl points
        locs = wpp3.all_point_finder(xf = xf, zf = zf, d1_xf = d1_xf, d1_zf = d1_zf, d2_xf = d2_xf, 
                                d2_zf = d2_zf, max_it = max_it, prec_it = prec_it, loc_threshhold = loc_threshhold, 
                                point_itnum = point_itnum, deg_treshhold = deg_treshhold)
        #print(locs)
        #get the carges
        charge_list = wpp3.weyl_charge_calculator(d1_xf = d1_xf, d1_zf = d1_zf, d2_xf = d2_xf, d2_zf = d2_zf, locs = locs)
        
        #append these to the containers
        locs_t.append(locs)
        charges_t.append(charge_list)
        
    #then return these containers
    return locs_t, charges_t