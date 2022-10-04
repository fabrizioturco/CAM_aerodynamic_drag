"""Separation distance module.

This module calculates achievable separation distances.

Example
-------


Notes
-----


Attributes
----------


"""

import numpy as np
from math import ceil
from datetime import datetime,timedelta
import astropy
from astropy import units as u
from warnings import warn

from constants import GM_Earth


def calc_separation_distance(   CB : u.Quantity,
                                CB_ref : u.Quantity,
                                dens : u.Quantity, 
                                semi_major_axis : u.Quantity, 
                                ts : np.ndarray,
                                tc : timedelta=None):
    """
    Calculates the separation distance for a given density, depending on 
    orbit, reference and actual ballistic coefficient and manoeuvring time 
    using the analytic equation by Omar/Bevilacqua.

    Parameters
    ----------
    CB : u.Quantity
        Actual ballistic coefficient.
    CB_ref : u.Quantity
        Reference ballistic coefficient.
    dens : u.Quantity
        Atmospheric density.
    semi_major_axis : u.Quantity
        Semi-major axis of the satellite's orbit.
    ts : u.Quantity
        Manoeuvring time, i.e., time spent in non-reference configuration.
    tc : u.Quantity
        Time spent in reference configuration.

    Returns
    -------
    miss_dist : u.Quantity
        Miss distance for manoeuvring times in ts.

    Raises
    ------
    AttributeError
        The ``Raises`` section is a list of all exceptions
        that are relevant to the interface.
    ValueError
        If `param2` is equal to `param1`.

    """
    if tc is None:
        tc = ts.to(u.s).value
    else:
        tc = tc.total_seconds()
    
    # Unit conversions
    dens = dens.to(u.kg/u.m**3).value
    muE = GM_Earth.to(u.m**3/u.s**2).value
    semi_major_axis = semi_major_axis.to(u.m).value
    CB = CB.to(u.m**2/u.kg).value
    CB_ref = CB_ref.to(u.m**2/u.kg).value
    ts = ts.to(u.s).value

    phi_ddot = 3*dens*muE/(2*semi_major_axis**2) * (CB - CB_ref)
    phi_dot = phi_ddot *ts
    delta_phi = 1/2*phi_ddot*ts**2 + phi_ddot*ts*(tc-ts)
    delta_x = semi_major_axis*delta_phi

    return delta_x * u.m, delta_phi, phi_dot

def calc_separation_distance_constrained(   CB_1 : u.Quantity,
                                            CB_2 : u.Quantity,
                                            CB_ref : u.Quantity,
                                            dens : u.Quantity, 
                                            semi_major_axis : u.Quantity, 
                                            tsteps : np.ndarray,
                                            t_1 : timedelta,
                                            t_2 : timedelta):
    """
    Calculates the separation distance for a given density, depending on 
    orbit, reference and actual ballistic coefficient and manoeuvring time 
    using the analytic equation by Omar/Bevilacqua.

    Parameters
    ----------
    CB : u.Quantity
        Actual ballistic coefficient.
    CB_ref : u.Quantity
        Reference ballistic coefficient.
    dens : u.Quantity
        Atmospheric density.
    semi_major_axis : u.Quantity
        Semi-major axis of the satellite's orbit.
    ts : u.Quantity
        Manoeuvring time, i.e., time spent in non-reference configuration.
    tc : u.Quantity
        Time spent in reference configuration.

    Returns
    -------
    miss_dist : u.Quantity
        Miss distance for manoeuvring times in ts.

    Raises
    ------
    AttributeError
        The ``Raises`` section is a list of all exceptions
        that are relevant to the interface.
    ValueError
        If `param2` is equal to `param1`.

    """
   
    # Unit conversions
    dens = dens.to(u.kg/u.m**3).value
    muE = GM_Earth.to(u.m**3/u.s**2).value
    semi_major_axis = semi_major_axis.to(u.m).value
    CB_1 = CB_1.to(u.m**2/u.kg).value
    CB_2 = CB_2.to(u.m**2/u.kg).value
    CB_ref = CB_ref.to(u.m**2/u.kg).value
    tsteps = tsteps.to(u.s).value
    t_1 = t_1.total_seconds()
    t_2 = t_2.total_seconds()

    phi_ddot = np.zeros(len(tsteps))
    phi_dot = np.zeros(len(tsteps))
    phi = np.zeros(len(tsteps))

    phi_dot_int = [0]
    phi_int = [0]

    phase_duration = t_1+t_2

    num_phases = ceil(tsteps[-1]/phase_duration)

    t_end_1 = 0
    t_end_2 = 0
    
    phase = 1
    phi_dot_end_2 = 0
    phi_end_2 = 0
    for idx, t in enumerate(tsteps):
        if t > phase*phase_duration:
            phase = phase + 1

                
        if t <= t_1 + (phase-1)*phase_duration:
            tt = t - t_end_2
            phi_ddot[idx] = 3*dens[idx]*muE/(2*semi_major_axis**2) * (CB_1 - CB_ref)
            phi_dot[idx] = phi_ddot[idx]*tt + phi_dot_end_2
            phi[idx] = 1/2*phi_ddot[idx]*tt**2 + phi_dot_end_2*tt + phi_end_2
            phi_dot_end_1 = phi_dot[idx]
            phi_end_1 = phi[idx]
            t_end_1 = t
        # elif t <= phase*phase_duration:
        else:
            tt = t - t_end_1
            phi_ddot[idx] = 3*dens[idx]*muE/(2*semi_major_axis**2) * (CB_2 - CB_ref)
            phi_dot[idx] = phi_ddot[idx]*tt + phi_dot_end_1
            phi[idx] = 1/2*phi_ddot[idx]*tt**2 + phi_dot_end_1*tt + phi_end_1
            phi_dot_end_2 = phi_dot[idx]
            phi_end_2 = phi[idx]
            t_end_2 = t

    delta_x = semi_major_axis*phi

    return delta_x * u.m, phi, phi_dot, phi_ddot