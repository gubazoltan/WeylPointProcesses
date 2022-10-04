import numpy as np
import sympy as sp
import wpp2

def interpolator(x1, z1, x2, z2, beta1, beta2):
    
    #defien a new symbol,t, which characterises the interpolation between the two systems
    t = sp.Symbol('t')
    
    #define the interpolating expressions
    xt = (1 - t) * x1 + t * x2  
    zt = (1 - t) * z1 + t * z2
    
    #find the derivatives for these symbolic functions
    #find the symbolic derivatives of the components
    d1_xt = sp.diff(xt, beta1)
    d1_zt = sp.diff(zt, beta1)
    d2_xt = sp.diff(xt, beta2)
    d2_zt = sp.diff(zt, beta2)
    
    #return all the stuff we need
    return xt, zt, d1_xt, d1_zt, d2_xt, d2_zt, t

def three_point_process(x1, z1, x2, z2, beta1, beta2, ts, max_it = 30, prec_it = 1e-12, loc_threshhold = 1e-10,
                     point_itnum = 500, deg_treshhold = 1e-10):
    
    #define the interpolating system
    xt, zt, d1_xt, d1_zt, d2_xt, d2_zt, t = interpolator(x1 = x1, z1 = z1, x2 = x2, z2 = z2, beta1 = beta1, beta2 = beta2)
    
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
        xf = sp.utilities.lambdify(['beta_1', 'beta_2'], x)
        zf = sp.utilities.lambdify(['beta_1', 'beta_2'], z)
        d1_xf = sp.utilities.lambdify(['beta_1', 'beta_2'], d1_x)
        d1_zf = sp.utilities.lambdify(['beta_1', 'beta_2'], d1_z)
        d2_xf = sp.utilities.lambdify(['beta_1', 'beta_2'], d2_x)
        d2_zf = sp.utilities.lambdify(['beta_1', 'beta_2'], d2_z)
            
        #get the location of the weyl points
        locs = wpp2.all_point_finder(xf = xf, zf = zf, d1_xf = d1_xf, d1_zf = d1_zf, d2_xf = d2_xf, 
                                d2_zf = d2_zf, max_it = max_it, prec_it = prec_it, loc_threshhold = loc_threshhold, 
                                point_itnum = point_itnum, deg_treshhold = deg_treshhold)
        #print(locs)
        #get the carges
        charge_list = wpp2.weyl_charge_calculator(d1_xf = d1_xf, d1_zf = d1_zf, d2_xf = d2_xf, d2_zf = d2_zf, locs = locs)
        
        #append these to the containers
        locs_t.append(locs)
        charges_t.append(charge_list)
        
    #then return these containers
    return locs_t, charges_t