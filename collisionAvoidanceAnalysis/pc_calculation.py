"""Collision probability calculation module.

This module calcualtes collision probabilities with different methods

Example
-------

Notes
-----

Attributes
----------

"""

import numpy as np
from datetime import datetime, timedelta
from astropy import units as u
from scipy.integrate import quad, dblquad, nquad
from scipy.optimize import minimize, basinhopping, differential_evolution
from math import erf, log

omega_E = np.array([0,0,7.292115e-5])

# TODO: add @njit
def calculate_pc_chen(x1, v1, R1, C1, x2, v2, R2, C2, epoch, debug: bool = False):
    # unit conversions
    x1 = x1.to(u.m).value
    v1 = v1.to(u.m/u.s).value
    R1 = R1.to(u.m).value
    C1 = C1.to(u.m**2).value
    x2 = x2.to(u.m).value
    v2 = v2.to(u.m/u.s).value
    R2 = R2.to(u.m).value
    C2 = C2.to(u.m**2).value

    if debug: 
        print('x1 [m]:',x1)
        print('v1 [m/s]:',v1)
        print('C1 [m**2]:',C1)
        print('R1 [m]:',R1)
        print('x2 [m]:',x2)
        print('v2 [m/s]:',v2)
        print('C2 [m**2]:',C2)
        print('R2 [m]:',R2)

    # transformation to inertial velocities
    v1 = v1 + np.cross(omega_E,x1)
    v2 = v2 + np.cross(omega_E,x2)

    # get closest approach 
    tca, norm_r_tca, x1, x2 = get_closest_approach(x1, v1, x2, v2)

    # combined collision radius
    R_C = R1 + R2

    eta = np.linalg.norm(v2)/np.linalg.norm(v1)
    psi = np.arccos((v1 @ v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
    if debug: print('psi [deg]:',psi*180/np.pi)

    # transformation matrix from ITRF to NTW of object 1
    U = x1/np.linalg.norm(x1)
    W = np.cross(x1,v1)/np.linalg.norm(np.cross(x1,v1))
    V = np.cross(W,U)

    R_UVW_1 = np.array([U.T, V.T, W.T])
    if debug: print('R_UVW_1:',R_UVW_1)

    sigma_R = np.sqrt(C1[0,0] + C2[0,0])
    sigma_S = np.sqrt(C1[1,1] + C2[1,1])
    sigma_W = np.sqrt(C1[2,2] + C2[2,2])
    
    sigma_SW = np.sqrt(sigma_S**2*np.cos(psi/2)**2 + sigma_W**2*np.sin(psi/2)**2)

    sigma_X = sigma_R
    sigma_Y = sigma_SW
    if debug: print('sigma_X:',sigma_X)
    if debug: print('sigma_Y:',sigma_Y)

    # relative position and velocity in ITRF
    delta_r = np.array(x2 - x1)
    if debug: print('delta_r:',delta_r)
    # relative position and velocity in RTN of object 1
    delta_r_RTN1 = R_UVW_1@delta_r
    if debug: print('delta_r_RTN1:',delta_r_RTN1)

    R = delta_r_RTN1[0]
    S = delta_r_RTN1[1]
    W = delta_r_RTN1[2]
    if debug: print('R:',R)
    if debug: print('S:',S)
    if debug: print('W:',W)

    # final calculation of Pc
    Pc = np.exp(-0.5*(R**2/sigma_X**2 + (S**2 + W**2)/sigma_Y**2)) * (1-np.exp(-R_C**2/(2*sigma_X*sigma_Y)))

    return Pc, norm_r_tca


def calculate_pc_recursive(x1, v1, R1, C1, x2, v2, R2, C2, epoch, debug: bool = False):
    # unit conversions
    x1 = x1.to(u.m).value
    v1 = v1.to(u.m/u.s).value
    R1 = R1.to(u.m).value
    C1 = C1.to(u.m**2).value
    x2 = x2.to(u.m).value
    v2 = v2.to(u.m/u.s).value
    R2 = R2.to(u.m).value
    C2 = C2.to(u.m**2).value

    if debug: 
        print('x1 [m]:',x1)
        print('v1 [m/s]:',v1)
        print('C1 [m**2]:',C1)
        print('R1 [m]:',R1)
        print('x2 [m]:',x2)
        print('v2 [m/s]:',v2)
        print('C2 [m**2]:',C2)
        print('R2 [m]:',R2)

    # transformation to inertial velocities
    v1 = v1 + np.cross(omega_E,x1)
    v2 = v2 + np.cross(omega_E,x2)

    # combined collision radius
    R_C = R1 + R2

    # get closest approach 
    tca, norm_r_tca, x1, x2 = get_closest_approach(x1, v1, x2, v2, debug=debug)

    eta = np.linalg.norm(v2)/np.linalg.norm(v1)
    psi = np.arccos((v1 @ v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
    if debug: print('psi [deg]:',psi*180/np.pi)

    # transformation matrix from ITRF to NTW of object 1
    U = x1/np.linalg.norm(x1)
    W = np.cross(x1,v1)/np.linalg.norm(np.cross(x1,v1))
    V = np.cross(W,U)

    R_UVW_1 = np.array([U.T, V.T, W.T])
    if debug: print('R_UVW_1:',R_UVW_1)

    sigma_R = np.sqrt(C1[0,0] + C2[0,0])
    sigma_S = np.sqrt(C1[1,1] + C2[1,1])
    sigma_W = np.sqrt(C1[2,2] + C2[2,2])
    
    sigma_SW = np.sqrt(sigma_S**2*np.cos(psi/2)**2 + sigma_W**2*np.sin(psi/2)**2)

    sigma_X = sigma_R
    sigma_Y = sigma_SW
    if debug: print('sigma_X:',sigma_X)
    if debug: print('sigma_Y:',sigma_Y)

    # relative position and velocity in ITRF
    delta_r = np.array(x2 - x1)
    if debug: print('delta_r:',delta_r)
    # relative position and velocity in RTN of object 1
    delta_r_RTN1 = R_UVW_1@delta_r
    if debug: print('delta_r_RTN1:',delta_r_RTN1)

    R = delta_r_RTN1[0]
    S = delta_r_RTN1[1]
    W = delta_r_RTN1[2]
    if debug: print('R:',R)
    if debug: print('S:',S)
    if debug: print('W:',W)
    mu_X = R
    mu_Y = np.sqrt(S**2 + W**2)# delta_r_B = np.array([mu_X, mu_Y]
    delta_r_B = np.array([mu_X,mu_Y])
    if debug: print('delta_r_B:',delta_r_B)


    # final calculation of Pc
    v = 0.5*(delta_r_B[0]**2/sigma_X**2 + delta_r_B[1]**2/sigma_Y**2)
    u = R_C**2/(2*sigma_X*sigma_Y)
    P0 = np.exp(-v)*(1-np.exp(-u))
    P1 = v*P0 - u*v*np.exp(-(v+u))
    Pc = P0 + P1

    return Pc


def calculate_pc_foster(x1, v1, R1, C1, x2, v2, R2, C2, epoch, debug: bool = False):
    # unit conversions
    x1 = x1.to(u.m).value
    v1 = v1.to(u.m/u.s).value
    R1 = R1.to(u.m).value
    C1 = C1.to(u.m**2).value
    x2 = x2.to(u.m).value
    v2 = v2.to(u.m/u.s).value
    R2 = R2.to(u.m).value
    C2 = C2.to(u.m**2).value

    if debug: 
        print('x1 [m]:',x1)
        print('v1 [m/s]:',v1)
        print('C1 [m**2]:',C1)
        print('R1 [m]:',R1)
        print('x2 [m]:',x2)
        print('v2 [m/s]:',v2)
        print('C2 [m**2]:',C2)
        print('R2 [m]:',R2)

    # transformation to inertial velocities
    v1 = v1 + np.cross(omega_E,x1)
    v2 = v2 + np.cross(omega_E,x2)

    # get closest approach 
    tca, norm_r_tca, x1, x2 = get_closest_approach(x1, v1, x2, v2, debug=debug)

    # combined collision radius
    R_C = R1 + R2

    eta = np.linalg.norm(v2)/np.linalg.norm(v1)
    psi = np.arccos((v1 @ v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
    if debug: print('psi [deg]:',psi*180/np.pi)

    delta_r = np.array(x2 - x1)
    delta_v = np.array(v2 - v1)

    # transformation matrix from ITRF to NTW of object 1
    U = x1/np.linalg.norm(x1)
    W = np.cross(x1,v1)/np.linalg.norm(np.cross(x1,v1))
    V = np.cross(W,U)

    R_UVW_1 = np.array([U.T, V.T, W.T])
    if debug: print('R_UVW_1:',R_UVW_1)

    # transformation matrix from ITRF to NTW of object 2
    U = x2/np.linalg.norm(x2)
    W = np.cross(x2,v2)/np.linalg.norm(np.cross(x2,v2))
    V = np.cross(W,U)

    R_UVW_2 = np.array([U.T, V.T, W.T])
    if debug: print('R_UVW_2:',R_UVW_2)

    # transformation matrix from ITRF to B-plane
    X_B = delta_r/np.linalg.norm(delta_r)
    Y_B = np.cross(delta_r,delta_v)/np.linalg.norm(np.cross(delta_r,delta_v))

    R_XBYB = np.array([X_B.T, Y_B.T])

    C_tot = R_UVW_1.T@C1@R_UVW_1 + R_UVW_2.T@C2@R_UVW_2
    C_B = R_XBYB@C_tot@R_XBYB.T
    if debug: print('C_B:',C_B)
    C_B_inv = np.linalg.inv(C_B)

    # relative position and velocity in ITRF
    if debug: print('delta_r:',delta_r)
    delta_r_B = R_XBYB@delta_r
    if debug: print('delta_r_B:',delta_r_B)

    ## 2D-equation
    def f(y,x):
        A_B = 0.5*(delta_r_B+np.array([x,y])).T@ C_B_inv@(delta_r_B+np.array([x,y]))
        return np.exp(-A_B)

    def bounds_y(x):
        return [-np.sqrt(R_C**2-(x)**2), +np.sqrt(R_C**2-(x)**2)]
        # return [-R_C, +R_C]

    def bounds_x():
        return [-R_C, +R_C]

    integral = nquad(f, [bounds_y, bounds_x])

    Pc = 1/(2*np.pi*np.sqrt(np.linalg.det(C_B))) * integral[0]

    return Pc, C_B, delta_r_B, epoch+timedelta(seconds=tca)


def calculate_pc_max(x1, v1, R1, C1, x2, v2, R2, C2, epoch, debug: bool = False):
    # unit conversions
    x1 = x1.to(u.m).value
    v1 = v1.to(u.m/u.s).value
    R1 = R1.to(u.m).value
    C1 = C1.to(u.m**2).value
    x2 = x2.to(u.m).value
    v2 = v2.to(u.m/u.s).value
    R2 = R2.to(u.m).value
    C2 = C2.to(u.m**2).value

    if debug: 
        print('x1 [m]:',x1)
        print('v1 [m/s]:',v1)
        print('C1 [m**2]:',C1)
        print('R1 [m]:',R1)
        print('x2 [m]:',x2)
        print('v2 [m/s]:',v2)
        print('C2 [m**2]:',C2)
        print('R2 [m]:',R2)

    # transformation to inertial velocities
    v1 = v1 + np.cross(omega_E,x1)
    v2 = v2 + np.cross(omega_E,x2)

    # get closest approach 
    tca, norm_r_tca, x1, x2 = get_closest_approach(x1, v1, x2, v2, debug=debug)

    # combined collision radius
    R_C = R1 + R2

    eta = np.linalg.norm(v2)/np.linalg.norm(v1)
    psi = np.arccos((v1 @ v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
    if debug: print('psi [deg]:',psi*180/np.pi)

    delta_r = np.array(x2 - x1)
    if debug: print('delta_r:',delta_r)
    delta_v = np.array(v2 - v1)
    if debug: print('delta_v:',delta_v)

    # transformation matrix from ITRF to NTW of object 1
    U = x1/np.linalg.norm(x1)
    W = np.cross(x1,v1)/np.linalg.norm(np.cross(x1,v1))
    V = np.cross(W,U)

    R_UVW_1 = np.array([U.T, V.T, W.T])
    if debug: print('R_UVW_1:',R_UVW_1)

    # transformation matrix from ITRF to NTW of object 2
    U = x2/np.linalg.norm(x2)
    W = np.cross(x2,v2)/np.linalg.norm(np.cross(x2,v2))
    V = np.cross(W,U)

    R_UVW_2 = np.array([U.T, V.T, W.T])
    if debug: print('R_UVW_2:',R_UVW_2)

    # transformation matrix from ITRF to B-plane
    X_B = delta_r/np.linalg.norm(delta_r)
    Y_B = np.cross(delta_r,delta_v)/np.linalg.norm(np.cross(delta_r,delta_v))

    R_XBYB = np.array([X_B.T, Y_B.T])

    C_tot = R_UVW_1.T@C1@R_UVW_1 + R_UVW_2.T@C2@R_UVW_2
    C_B = R_XBYB@C_tot@R_XBYB.T
    if debug: print('C_B:',C_B)
    C_B_inv = np.linalg.inv(C_B)

    
    delta_r_B = R_XBYB@delta_r
    if debug: print('delta_r_B:',delta_r_B)

    # sigma_R = np.sqrt(C1[0,0] + C2[0,0])
    # sigma_S = np.sqrt(C1[1,1] + C2[1,1])
    # sigma_W = np.sqrt(C1[2,2] + C2[2,2])
    
    # sigma_SW = np.sqrt(sigma_S**2*np.cos(psi/2)**2 + sigma_W**2*np.sin(psi/2)**2)

    # sigma_X = sigma_R
    # sigma_Y = sigma_SW
    # if debug: print('sigma_X:',sigma_X)
    # if debug: print('sigma_Y:',sigma_Y)

    # C_B = np.diag([sigma_X**2, sigma_Y**2])
    # C_B_inv = np.linalg.inv(C_B)
    # if debug: print('C_B:',C_B)

    # # relative position and velocity in ITRF
    # delta_r = np.array(x2 - x1)
    # if debug: print('delta_r:',delta_r)
    # # relative position and velocity in RTN of object 1
    # delta_r_RTN1 = R_UVW_1@delta_r
    # if debug: print('delta_r_RTN1:',delta_r_RTN1)

    # R = delta_r_RTN1[0]
    # S = delta_r_RTN1[1]
    # W = delta_r_RTN1[2]
    # if debug: print('R:',R)
    # if debug: print('S:',S)
    # if debug: print('W:',W)
    # mu_X = R
    # mu_Y = np.sqrt(S**2 + W**2)# delta_r_B = np.array([mu_X, mu_Y]
    # delta_r_B = np.array([mu_X,mu_Y])
    # if debug: print('delta_r_B:',delta_r_B)

    # final calculation of Pc_max
    Pc_max = R_C**2 / ( np.exp(1)*np.sqrt(np.linalg.det(C_B))*(delta_r_B.T@C_B_inv@delta_r_B) )
    k = np.sqrt(0.5*delta_r_B.T@C_B_inv@delta_r_B)

    return Pc_max, k

def calculate_pc_max_alfano(x1, v1, R1, x2, v2, R2, AR=None, debug: bool = False):
    # unit conversions
    x1 = x1.to(u.m).value
    v1 = v1.to(u.m/u.s).value
    R1 = R1.to(u.m).value
    x2 = x2.to(u.m).value
    v2 = v2.to(u.m/u.s).value
    R2 = R2.to(u.m).value

    if debug:
        print('x1 [m]:',x1)
        print('R1 [m]:',R1)
        print('x2 [m]:',x2)
        print('R2 [m]:',R2)
        if AR is not None:
            print('AR [-]:',AR)
    
    # transformation to inertial velocities
    v1 = v1 + np.cross(omega_E,x1)
    v2 = v2 + np.cross(omega_E,x2)

    # get closest approach 
    tca, norm_r_tca, x1, x2 = get_closest_approach(x1, v1, x2, v2, debug=debug)

    r_miss = np.linalg.norm(x2-x1)
    R = R1 + R2
    r_sd = R/r_miss

    if AR is not None:
        sigma = r_miss/(np.sqrt(2)*AR)
        Pc_max = R**2/(2*AR*sigma**2) * np.exp(-1/2*(r_miss/(AR*sigma))**2)
    
    if AR is None:
        print('Assuming AR=inf.')
        Pc_max = 1/2 * (  erf( (r_sd+1)/(2*np.sqrt(r_sd)) * np.sqrt(-log((1-r_sd)/(1+r_sd))) )
                + erf( (r_sd-1)/(2*np.sqrt(r_sd)) * np.sqrt(-log((1-r_sd)/(1+r_sd))) )  )

    return Pc_max

def calculate_pc_max_scaled(x1, v1, R1, C1, x2, v2, R2, C2, epoch, debug: bool = False):
    # unit conversions
    x1 = x1.to(u.m).value
    v1 = v1.to(u.m/u.s).value
    R1 = R1.to(u.m).value
    C1 = C1.to(u.m**2).value
    x2 = x2.to(u.m).value
    v2 = v2.to(u.m/u.s).value
    R2 = R2.to(u.m).value
    C2 = C2.to(u.m**2).value

    if debug: 
        print('x1 [m]:',x1)
        print('v1 [m/s]:',v1)
        print('C1 [m**2]:',C1)
        print('R1 [m]:',R1)
        print('x2 [m]:',x2)
        print('v2 [m/s]:',v2)
        print('C2 [m**2]:',C2)
        print('R2 [m]:',R2)

    # transformation to inertial velocities
    v1 = v1 + np.cross(omega_E,x1)
    v2 = v2 + np.cross(omega_E,x2)

    # get closest approach 
    tca, norm_r_tca, x1, x2 = get_closest_approach(x1, v1, x2, v2, debug=debug)

    # combined collision radius
    R_C = R1 + R2

    eta = np.linalg.norm(v2)/np.linalg.norm(v1)
    psi = np.arccos((v1 @ v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
    if debug: print('psi [deg]:',psi*180/np.pi)

    delta_r = np.array(x2 - x1)
    if debug: print('delta_r:',delta_r)
    delta_v = np.array(v2 - v1)
    if debug: print('delta_v:',delta_v)

    # transformation matrix from ITRF to NTW of object 1
    U = x1/np.linalg.norm(x1)
    W = np.cross(x1,v1)/np.linalg.norm(np.cross(x1,v1))
    V = np.cross(W,U)

    R_UVW_1 = np.array([U.T, V.T, W.T])
    if debug: print('R_UVW_1:',R_UVW_1)

    # transformation matrix from ITRF to NTW of object 2
    U = x2/np.linalg.norm(x2)
    W = np.cross(x2,v2)/np.linalg.norm(np.cross(x2,v2))
    V = np.cross(W,U)

    R_UVW_2 = np.array([U.T, V.T, W.T])
    if debug: print('R_UVW_2:',R_UVW_2)

    # transformation matrix from ITRF to B-plane
    X_B = delta_r/np.linalg.norm(delta_r)
    Y_B = np.cross(delta_r,delta_v)/np.linalg.norm(np.cross(delta_r,delta_v))

    R_XBYB = np.array([X_B.T, Y_B.T])
    
    

    delta_r_B = R_XBYB@delta_r
    if debug: print('delta_r_B:',delta_r_B)

    def pc_fun(k):
        # C_tot = k[0]**2*R_UVW_1.T@C1@R_UVW_1 + k[1]**2*R_UVW_2.T@C2@R_UVW_2
        # C = k**2 * C_tot
        C_tot = R_UVW_1.T@np.diag(k[:3])**2*C1@R_UVW_1 + R_UVW_2.T@np.diag(k[3:])**2*C2@R_UVW_2

        C_B = R_XBYB@C_tot@R_XBYB.T
        C_B_inv = np.linalg.inv(C_B)
        if np.linalg.det(C_B)<0:
            return 0
        ## 2D-equation
        def f(y,x):
            A_B = 0.5*(delta_r_B+np.array([x,y])).T@ C_B_inv@(delta_r_B+np.array([x,y]))
            return np.exp(-A_B)

        def bounds_y(x):
            return [-np.sqrt(R_C**2-(x)**2), +np.sqrt(R_C**2-(x)**2)]
            # return [-R_C, +R_C]

        def bounds_x():
            return [-R_C, +R_C]

        integral = nquad(f, [bounds_y, bounds_x])

        Pc = 1/(2*np.pi*np.sqrt(np.linalg.det(C_B))) * integral[0]
        return -Pc
    
    bnds = ((0, 5), (0, 5), (0, 5), (0, 5), (0, 5), (0, 5))
    # bnds = ((0, None))
    res = differential_evolution(pc_fun, x0=[1,1,1,1,1,1], bounds=bnds)
    # print(res)
    Pc_max = -res.fun
    k = res.x

    return Pc_max, k


def get_closest_approach(x1, v1, x2, v2, debug: bool = False):
    """
    This function calculates the closest approach between two objects
    during a close encounter under the assumption of constant linear motion.

    Parameters
    ----------
    x1 : np.ndarray
    v1 : np.ndarray
    x2 : np.ndarray
    v2 : np.ndarray
    debug : bool

    Returns
    -------
    tca : float
    norm_r_tca : float
    r1_tca : np.ndarray
    r2_tca : np.ndarray

    Raises
    ------

    """
    def dist(t):
        r1 = x1 + t*v1
        r2 = x2 + t*v2
        d = np.linalg.norm(r2-r1)
        return d
    
    res = minimize(dist, x0=0)
    tca = res.x[0]
    if debug: print('TCA:',tca)

    r1_tca = x1 + tca*v1
    r2_tca = x2 + tca*v2
    norm_r_tca = dist(tca)

    return tca, norm_r_tca, r1_tca, r2_tca