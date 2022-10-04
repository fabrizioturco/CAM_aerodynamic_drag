"""Aerodynamics module.

This module determines the ballistic coefficient of the Flying Laptop depending
on solar and geomagnetic activity based on loaded look-up tables.

Example
-------

Notes
-----

Attributes
----------

"""


import numpy as np
from scipy.interpolate import interpn
import astropy.units as u


CDmax_Ap0 = np.loadtxt('aero_data/CDmax_Ap0.txt', delimiter=',')
CDmax_Ap5 = np.loadtxt('aero_data/CDmax_Ap5.txt', delimiter=',')
CDmax_Ap10 = np.loadtxt('aero_data/CDmax_Ap10.txt', delimiter=',')
CDmax_Ap15 = np.loadtxt('aero_data/CDmax_Ap15.txt', delimiter=',')
CDmax_Ap20 = np.loadtxt('aero_data/CDmax_Ap20.txt', delimiter=',')
CDmax_Ap25 = np.loadtxt('aero_data/CDmax_Ap25.txt', delimiter=',')
CDmax_Ap30 = np.loadtxt('aero_data/CDmax_Ap30.txt', delimiter=',')
CDmax_Ap35 = np.loadtxt('aero_data/CDmax_Ap35.txt', delimiter=',')
CDmax_Ap40 = np.loadtxt('aero_data/CDmax_Ap40.txt', delimiter=',')
CDmax_Ap45 = np.loadtxt('aero_data/CDmax_Ap45.txt', delimiter=',')
CDnadir_Ap0 = np.loadtxt('aero_data/CDnadir_Ap0.txt', delimiter=',')
CDnadir_Ap5 = np.loadtxt('aero_data/CDnadir_Ap5.txt', delimiter=',')
CDnadir_Ap10 = np.loadtxt('aero_data/CDnadir_Ap10.txt', delimiter=',')
CDnadir_Ap15 = np.loadtxt('aero_data/CDnadir_Ap15.txt', delimiter=',')
CDnadir_Ap20 = np.loadtxt('aero_data/CDnadir_Ap20.txt', delimiter=',')
CDnadir_Ap25 = np.loadtxt('aero_data/CDnadir_Ap25.txt', delimiter=',')
CDnadir_Ap30 = np.loadtxt('aero_data/CDnadir_Ap30.txt', delimiter=',')
CDnadir_Ap35 = np.loadtxt('aero_data/CDnadir_Ap35.txt', delimiter=',')
CDnadir_Ap40 = np.loadtxt('aero_data/CDnadir_Ap40.txt', delimiter=',')
CDnadir_Ap45 = np.loadtxt('aero_data/CDnadir_Ap45.txt', delimiter=',')
CDmin_Ap0 = np.loadtxt('aero_data/CDmin_Ap0.txt', delimiter=',')
CDmin_Ap5 = np.loadtxt('aero_data/CDmin_Ap5.txt', delimiter=',')
CDmin_Ap10 = np.loadtxt('aero_data/CDmin_Ap10.txt', delimiter=',')
CDmin_Ap15 = np.loadtxt('aero_data/CDmin_Ap15.txt', delimiter=',')
CDmin_Ap20 = np.loadtxt('aero_data/CDmin_Ap20.txt', delimiter=',')
CDmin_Ap25 = np.loadtxt('aero_data/CDmin_Ap25.txt', delimiter=',')
CDmin_Ap30 = np.loadtxt('aero_data/CDmin_Ap30.txt', delimiter=',')
CDmin_Ap35 = np.loadtxt('aero_data/CDmin_Ap35.txt', delimiter=',')
CDmin_Ap40 = np.loadtxt('aero_data/CDmin_Ap40.txt', delimiter=',')
CDmin_Ap45 = np.loadtxt('aero_data/CDmin_Ap45.txt', delimiter=',')


m = 108.8
A_ref = 2.182

ap_range = np.loadtxt('aero_data/Ap.txt')
f107_range = np.loadtxt('aero_data/F107.txt')
f107a_range = np.loadtxt('aero_data/F107A.txt')

points=(ap_range[:,0],f107_range,f107a_range)

z_max = A_ref/m * np.array([CDmax_Ap0, CDmax_Ap5, CDmax_Ap10, CDmax_Ap15, CDmax_Ap20, CDmax_Ap25, CDmax_Ap30, CDmax_Ap35, CDmax_Ap40, CDmax_Ap45])
z_nadir = A_ref/m * np.array([CDnadir_Ap0, CDnadir_Ap5, CDnadir_Ap10, CDnadir_Ap15, CDnadir_Ap20, CDnadir_Ap25, CDnadir_Ap30, CDnadir_Ap35, CDnadir_Ap40, CDnadir_Ap45])
z_min = A_ref/m * np.array([CDmin_Ap0, CDmin_Ap5, CDmin_Ap10, CDmin_Ap15, CDmin_Ap20, CDmin_Ap25, CDmin_Ap30, CDmin_Ap35, CDmin_Ap40, CDmin_Ap45])



def calc_CB( f107, f107a, ap):
    """
    This function determines the ballistic coefficient of the Flying Laptop depending
    on solar and geomagnetic activity based on loaded look-up tables.

    Parameters
    ----------
    f107 : float
        The solar flux at 10.7 cm, proxy for solar activity.
    f107a : float
        The 81-day centred average of the solar flux at 10.7 cm.
    ap : float
        The Ap value, index for geomagentic activity.

    Returns
    -------
    CB_max : u.Quantity
        Ballistic coefficient of the Flying Laptop in maximum drag attitude.
    CB_nadir : u.Quantity
        Ballistic coefficient of the Flying Laptop in naidr-pointing attitude.
    CB_min : u.Quantity
        Ballistic coefficient of the Flying Laptop in minimum drag attitude.

    Raises
    ------

    """
    point = (ap,f107,f107a)
    CB_max = interpn(points,z_max,point,bounds_error=False,fill_value=None)
    CB_nadir= interpn(points,z_nadir,point,bounds_error=False,fill_value=None)
    CB_min = interpn(points,z_min,point,bounds_error=False,fill_value=None)
    
    return CB_max[0] * u.m**2/u.kg, CB_nadir[0] * u.m**2/u.kg, CB_min[0] * u.m**2/u.kg