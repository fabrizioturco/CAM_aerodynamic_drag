import numpy as np
from datetime import datetime, timedelta, timezone
from astropy import units as u
from astropy.time import Time, TimeDelta, TimeDeltaDatetime, TimezoneInfo
from astropy import coordinates as coord
from astropy.coordinates import SkyCoord, ITRS, GCRS, TEME
from astropy.coordinates import solar_system_ephemeris
from astropy.visualization import time_support
from numba import njit
from poliastro.bodies import Sun, Earth, Moon
from poliastro.twobody import Orbit
from poliastro.core.propagation import func_twobody
from poliastro.twobody.propagation import propagate, cowell
from poliastro.constants import rho0_earth, H0_earth, GM_earth
from poliastro.earth.atmosphere.jacchia import Jacchia77
from poliastro.earth.atmosphere.coesa76 import COESA76
from poliastro.twobody.angles import nu_to_E
from poliastro.ephem import build_ephem_interpolant
from sgp4.api import Satrec
from sgp4.api import jday
from scipy.integrate import quad, dblquad, nquad, DOP853, RK45, solve_ivp
from scipy.integrate._ivp.common import OdeSolution
from skyfield.positionlib import ITRF_to_GCRS, ITRF_to_GCRS2
from skyfield.positionlib import Geocentric, ICRF
from skyfield.framelib import itrs
from skyfield.api import load, wgs84
from nrlmsise00 import msise_flat
from ccsds_ndm.ndm_io import NdmIo

from getPositionFromTle import GetPositionFromTle

skyfield_ts = load.timescale()
utc = TimezoneInfo(utc_offset=0*u.hour)

solar_system_ephemeris.set("de432s")


def eci2ecef(x, y, z, v_x, v_y, v_z, time):
    skyfield_time = skyfield_ts.from_datetime(time)
    position_au = np.array([(x*u.km).to(u.AU).value, (y*u.km).to(u.AU).value, (z*u.km).to(u.AU).value])
    velocity_au_per_d = np.array([(v_x*u.km/u.s).to(u.AU/u.d).value, (v_y*u.km/u.s).to(u.AU/u.d).value, (v_z*u.km/u.s).to(u.AU/u.d).value])
    posvel_eci = Geocentric(position_au, velocity_au_per_d=velocity_au_per_d , t=skyfield_time)
    r,v =  posvel_eci.frame_xyz_and_velocity(itrs)

    return r.km, v.km_per_s

def eci2ecef_latlon(x, y, z, v_x, v_y, v_z, time):
    # DO NOT USE
    skyfield_time = skyfield_ts.from_datetime(time)
    position_au = np.array([(x*u.km).to(u.AU).value, (y*u.km).to(u.AU).value, (z*u.km).to(u.AU).value])
    velocity_au_per_d = np.array([(v_x*u.km/u.s).to(u.AU/u.d).value, (v_y*u.km/u.s).to(u.AU/u.d).value, (v_z*u.km/u.s).to(u.AU/u.d).value])
    posvel_eci = Geocentric(position_au, velocity_au_per_d=velocity_au_per_d , t=skyfield_time)
    r,v =  posvel_eci.frame_xyz_and_velocity(itrs)
    lat, lon, alt = posvel_eci.frame_latlon(itrs)

    return r.km, v.km_per_s, lat, lon, alt

def eci2ecef_old(x, y, z, v_x, v_y, v_z, time):
    gps_eci = GCRS(x=x*u.m, y=y*u.m, z=z*u.m, v_x=v_x*u.m/u.s, v_y=v_y*u.m/u.s, v_z=v_z*u.m/u.s, representation_type='cartesian',differential_type='cartesian', obstime=Time(time))
    gps_ecef = gps_eci.transform_to(ITRS(obstime=Time(time)))
    gps_ecef.representation_type = 'cartesian'

    return [gps_ecef.x.value, gps_ecef.y.value, gps_ecef.z.value], [gps_ecef.v_x.value, gps_ecef.v_y.value, gps_ecef.v_z.value]


def ecef2eci(x, y, z, v_x, v_y, v_z, time):
    skyfield_time = skyfield_ts.from_datetime(time)
    r = ITRF_to_GCRS(skyfield_time, np.array([x,y,z]))
    v = ITRF_to_GCRS(skyfield_time, np.array([v_x,v_y,v_z]))

    return r, v*1e-3

def ecef2eci_old(x, y, z, v_x, v_y, v_z, time):
    gps_ecef = ITRS(x=x*u.m, y=y*u.m, z=z*u.m, v_x=v_x*u.m/u.s, v_y=v_y*u.m/u.s, v_z=v_z*u.m/u.s, representation_type='cartesian',differential_type='cartesian', obstime=Time(time))
    gps_eci = gps_ecef.transform_to(GCRS(obstime=Time(time)))
    gps_eci.representation_type = 'cartesian'

    return [gps_eci.x.value, gps_eci.y.value, gps_eci.z.value], [gps_eci.v_x.value, gps_eci.v_y.value, gps_eci.v_z.value]

def teme2eci(x, y, z, v_x, v_y, v_z, time):
    gps_teme = TEME(x=x*u.m, y=y*u.m, z=z*u.m, v_x=v_x*u.m/u.s, v_y=v_y*u.m/u.s, v_z=v_z*u.m/u.s, representation_type='cartesian',differential_type='cartesian', obstime=Time(time))
    gps_eci = gps_teme.transform_to(GCRS(obstime=Time(time)))
    gps_eci.representation_type = 'cartesian'

    return [gps_eci.x.value, gps_eci.y.value, gps_eci.z.value], [gps_eci.v_x.value, gps_eci.v_y.value, gps_eci.v_z.value]