"""Density modelling module.

This module determines average densities along a trajectory specified by TLEs.

Example
-------

Notes
-----

Attributes
----------


"""


import numpy as np
from datetime import datetime,timedelta
from astropy import units as u
from astropy.time import Time
from pyatmos import expo,coesa76,nrlmsise00,jb2008
from pyatmos.jb2008.spaceweather import download_sw_jb2008,read_sw_jb2008
from pyatmos.jb2008.spaceweather import get_sw as get_sw_jb2008
from os import path,makedirs,remove
from pathlib import Path
import requests
from tqdm import tqdm
from colorama import Fore
from time import sleep
from nrlmsise00 import msise_flat
from scipy.interpolate import interp2d,interpn


from getPositionFromTle import GetPositionFromTle
from constants import R_Earth


def calc_mean_density_sgp4( start_time : datetime,
                            ts : u.Quantity,
                            model: str='nrlmsise00',
                            TLE_folder : str='\TLEs',
                            info: bool=False,
                            alt_output: bool=False,
                            z: np.ndarray=None):
    """
    This function calculates the mean density by sampling a selected
    density model's output at positions gained by propagating a TLE.

    Parameters
    ----------
    start_time : datetime
        Start epoch of calculations.
    ts : u.Quantity
        Timesteps at which the density is to be evaluated.
    model : string
        The atmosphere model to be used.
    TLE_folder : str
        Folder in which TLEs are (to be) stored.
    alt_output : bool
        Flag for output of altitude.
    

    Returns
    -------
    dens_avg : u.Quantity
        Average density.
    dens : u.Quantity
        Density.
    f107_avg : float
        Averaged solar flux at 10.7 cm.
    f107a_avg : float
        Averaged 81-day centred mean of solar flux at 10.7 cm.
    ap_avg : float
        Averaged geomagentic activity index Ap.
    alt_out : u.Quantity
        Altitude.

    Raises
    ------

    """
    if model not in ['nrlmsise00','jb08','coesa76','atmden']:
        raise ValueError('Atmosphere model not supported.')
    
    # Unit conversions
    ts = ts.to(u.s).value

    # TLE handling
    gpfTLE = GetPositionFromTle(tle_file_path=TLE_folder, maximum_tle_age = timedelta(days=365*6))

    # variable initialization
    dens = np.zeros(len(ts))
    f107 = np.zeros(len(ts))
    f107a = np.zeros(len(ts))
    ap = np.zeros(len(ts))

    # space weather data
    swfile = download_sw_nrlmsise00('/sw-data/')
    sw_data= read_sw_nrlmsise00(swfile)

    if model=='jb08':
        swfile_jb = download_sw_jb2008('/sw-data/')
        if info: print(swfile_jb)
        sw_data_jb = read_sw_jb2008(swfile_jb)
        # F10,F10B,S10,S10B,M10,M10B,Y10,Y10B,DTCVAL = get_sw_jb2008(sw_data,float(start_time.strftime("mjd")))
        # if info: print(sw_data)
    elif model=='atmden':
        alt_range = np.array([550,600,700])
        lon_range = np.arange(-180,180,5)
        lat_range = np.arange(-85,90,5)
        points=(alt_range,lat_range,lon_range)

    if alt_output:
        alt_out = np.zeros(len(ts))

    # density evaluation
    for idx, val in enumerate(ts):
        time = start_time + timedelta(seconds=val)
        t = time.strftime('%Y-%m-%d %H:%M:%S')
        t_mjd = Time(time).mjd
        lat,lon,alt = gpfTLE.getLatLonHeightFromTle(time, warn_if_tle_differs_in_time = False)
        alt = alt*1e-3

        if model=='expo':
            res = expo(alt)
            dens[idx] = res.rho # kg/m**3
        elif model=='coesa76':
            res = coesa76(alt)
            dens[idx] = res.rho # kg/m**3
        elif model=='nrlmsise00':
            res = nrlmsise00(time,(lat,lon,alt),sw_data)
            dens[idx] = res.rho # kg/m**3
            _,f107[idx],ap[idx],_ = get_sw(sw_data,time.strftime("%Y%m%d"),0)
        elif model=='jb08':
            res = jb2008(t,(lat,lon,alt),sw_data_jb)
            dens[idx] = res.rho # kg/m**3
            f107a[idx],f107[idx],ap[idx],_ = get_sw(sw_data,time.strftime("%Y%m%d"),0)
            F10,F10B,S10,S10B,M10,M10B,Y10,Y10B,DTCVAL = get_sw_jb2008(sw_data_jb,t_mjd)
        elif model=='atmden':
            point = (alt,lat,lon)
            dens[idx] = interpn(points,z,point,bounds_error=False,fill_value=None) # kg/m**3

        if alt_output:
            alt_out[idx] = alt
    
    dens_avg = np.cumsum(dens[::-1]) / np.arange(1,len(dens)+1)
    f107_avg = np.mean(f107)
    f107a_avg = np.mean(f107a)
    ap_avg = np.mean(ap)

    if alt_output:
        return dens_avg * u.kg/u.m**3, dens * u.kg/u.m**3, f107_avg, f107a_avg, ap_avg, alt_out*u.km
    else:
        return dens_avg * u.kg/u.m**3, dens * u.kg/u.m**3, f107_avg, f107a_avg, ap_avg

def calc_mean_density_sgp4_activity(start_time : datetime,
                            ts : u.Quantity,
                            model: str='nrlmsise00',
                            TLE_folder : str='/TLEs',
                            info: bool=False,
                            alt_output: bool=False):
    """
    This function calculates the mean density for defined levels of solar and geomagnetic activity
    by sampling a selected density model's output at positions gained by propagating a TLE.

    Parameters
    ----------
    start_time : datetime
        Start epoch of calculations.
    ts : u.Quantity
        Timesteps at which the density is to be evaluated.
    model : string
        The atmosphere model to be used.
    TLE_folder : str
        Folder in which TLEs are (to be) stored.
    alt_output : bool
        Flag for output of altitude.

    Returns
    -------
    dens_avg_low : u.Quantity
        Average density for low solar and geomagnetic activity.
    dens_low : u.Quantity
        Density for low solar and geomagnetic activity.
    dens_avg_moderate : u.Quantity
        Average density for mdoerate solar and geomagnetic activity.
    dens_moderate : u.Quantity
        Density for moderate solar and geomagnetic activity.
    dens_avg_high : u.Quantity
        Average density for high solar and geomagnetic activity.
    dens_high : u.Quantity
        Density for high solar and geomagnetic activity.
    alt_out : u.Quantity
        Altitude.

    Raises
    ------


    """
    if model not in ['nrlmsise00','jb08','coesa76']:
        raise ValueError('Atmosphere model not supported.')
    
    # Unit conversions
    ts = ts.to(u.s).value

    # TLE handling
    gpfTLE = GetPositionFromTle(tle_file_path=TLE_folder, maximum_tle_age = timedelta(days=365*6))
    print(gpfTLE.get_tle_at(start_time))
    # variable initialization
    dens_low = np.zeros(len(ts))
    dens_moderate = np.zeros(len(ts))
    dens_high = np.zeros(len(ts))

    if alt_output:
        alt_out = np.zeros(len(ts))
        
    # density evaluation
    for idx, val in enumerate(ts):
        time = start_time + timedelta(seconds=val)
        t = time.strftime('%Y-%m-%d %H:%M:%S')
        lat,lon,alt = gpfTLE.getLatLonHeightFromTle(time, warn_if_tle_differs_in_time = False)
        alt = alt*1e-3

        if model=='nrlmsise00':
            res_low = msise_flat(time, alt, lat, lon, 65, 65, 0, method="gtd7d")
            res_moderate = msise_flat(time, alt, lat, lon, 140, 140, 15, method="gtd7d")
            res_high = msise_flat(time, alt, lat, lon, 250, 250, 45, method="gtd7d")
        else:
            return
        
        dens_low[idx] = (res_low[5]*u.g/u.cm**3).to(u.kg/u.m**3).value # kg/m**3
        dens_moderate[idx] = (res_moderate[5]*u.g/u.cm**3).to(u.kg/u.m**3).value # kg/m**3
        dens_high[idx] = (res_high[5]*u.g/u.cm**3).to(u.kg/u.m**3).value # kg/m**3


        if alt_output:
            alt_out[idx] = alt
    
    dens_avg_low = np.cumsum(dens_low) / np.arange(1,len(dens_low)+1)
    # print('dens_avg_low:',dens_avg_low)
    dens_avg_moderate = np.cumsum(dens_moderate) / np.arange(1,len(dens_moderate)+1)
    # print('dens_avg_moderate:',dens_avg_moderate)
    dens_avg_high = np.cumsum(dens_high) / np.arange(1,len(dens_high)+1)
    # print('dens_avg_high:',dens_avg_high)

    if alt_output:
        return dens_avg_low * u.kg/u.m**3, dens_low * u.kg/u.m**3, dens_avg_moderate * u.kg/u.m**3, dens_moderate * u.kg/u.m**3, dens_avg_high * u.kg/u.m**3, dens_high * u.kg/u.m**3, alt_out*u.km
    return dens_avg_low * u.kg/u.m**3, dens_low * u.kg/u.m**3, dens_avg_moderate * u.kg/u.m**3, dens_moderate * u.kg/u.m**3, dens_avg_high * u.kg/u.m**3, dens_high * u.kg/u.m**3

# the following is taken from the pyatmos package with slight modifications
def tqdm_request(url,dir_to,file,desc):
    '''
    Try to download files from a remote server by request with a colored progress bar.
    '''
    block_size = 1024*10
    bar_format = "{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET)
    for idownload in range(5):
        try:
            local_file = open(dir_to + file, 'ab')
            pos = local_file.tell()
            res = requests.get(url,stream=True,timeout=100,headers={'Accept-Encoding': None,'Range': f'bytes={pos}-'})
            total_size = int(res.headers.get('content-length'))
            pbar = tqdm(desc = desc,total=total_size,unit='B',unit_scale=True,bar_format = bar_format,position=0,initial=pos) 
            for chunk in res.iter_content(block_size):
                pbar.update(len(chunk))
                local_file.write(chunk)  
            pbar.close()  
            res.close()  
            break
        except: 
            sleep(2)
            if idownload == 4:
                remove(dir_to + file)
                print('No response, skip this file.') 
        finally:    
            local_file.close() 

def download_sw_nrlmsise00(direc=None):
    '''
    Download or update the space weather data from www.celestrak.com

    Usage: 
    swfile = download_sw([direc])

    Inputs: 
    direc -> [str, optional] Directory for storing the space weather data
    
    Outputs: 
    swfile -> [str] Path of the space weather data

    Examples:
    >>> swfile = download_sw()
    >>> swfile = download_sw('sw-data/')
    '''
    
    if direc is None:
        home = str(Path.home())
        direc = home + '/src/sw-data/'
    
    swfile = direc + 'SW-All.txt'
    url = 'https://www.celestrak.com/SpaceData/SW-All.txt'

    if not path.exists(direc): makedirs(direc)
    if not path.exists(swfile):
        desc = 'Downloading the space weather data {:s} from CELESTRAK'.format('SW-All.txt')
        tqdm_request(url,direc,'SW-All.txt',desc)
    else:
        modified_time = datetime.fromtimestamp(path.getmtime(swfile))
        if datetime.now() > modified_time + timedelta(days=1):
            remove(swfile)
            desc = 'Updating the space weather data {:s} from CELESTRAK'.format('SW-All.txt')
            tqdm_request(url,direc,'SW-All.txt',desc)   
        else:
            print('The space weather data in {:s} is already the latest.'.format(direc))   
    return swfile
def read_sw_nrlmsise00(swfile):
    '''
    Parse and read the space weather data

    Usage: 
    sw_obs_pre = read_sw(swfile)

    Inputs: 
    swfile -> [str] Path of the space weather data
    
    Outputs: 
    sw_obs_pre -> [2d str array] Content of the space weather data

    Examples:
    >>> swfile = 'sw-data/SW-All.txt'
    >>> sw_obs_pre = read_sw(swfile)
    >>> print(sw_obs_pre)
    [['2020' '01' '07' ... '72.4' '68.0' '71.0']
    ['2020' '01' '06' ... '72.4' '68.1' '70.9']
    ...
    ...
    ['1957' '10' '02' ... '253.3' '267.4' '231.7']
    ['1957' '10' '01' ... '269.3' '266.6' '230.9']]
    '''
    sw_data = open(swfile,'r').readlines()
    SW_OBS,SW_PRE = [],[]
    flag1 = flag2 = 0
    for line in sw_data:
        # print(line)
        if line.startswith('BEGIN OBSERVED'): 
            flag1 = 1
            continue
        if line.startswith('END OBSERVED'): flag1 = 0 
        if flag1 == 1: 
            sw_p = line.split()
            # if len(sw_p) == 30:
            #     del sw_p[24]
            # elif len(sw_p) == 31: 
            #     sw_p = np.delete(sw_p,[23,25]) 
            # else: 
            # sw_p = np.delete(sw_p,[24,25,27])
            F107_OBS = sw_p[30]
            F107_OBS_CTR81 = sw_p[31]
            F107_OBS_LST81 = sw_p[32]
            sw_p[30] = sw_p[26]
            sw_p[31] = sw_p[28]
            sw_p[32] = sw_p[29]
            sw_p[26] = F107_OBS
            sw_p[28] = F107_OBS_CTR81
            sw_p[29] = F107_OBS_LST81
            del sw_p[27]
            SW_OBS.append(sw_p)
            
        if line.startswith('BEGIN DAILY_PREDICTED'): 
            flag2 = 1
            continue    
        if line.startswith('END DAILY_PREDICTED'): break 
        if flag2 == 1: 
            sw_p = line.split()
            F107_OBS = sw_p[29]
            F107_OBS_CTR81 = sw_p[30]
            F107_OBS_LST81 = sw_p[31]
            sw_p[29] = sw_p[26]
            sw_p[30] = sw_p[27]
            sw_p[31] = sw_p[27]
            sw_p[26] = F107_OBS
            sw_p[27] = F107_OBS_CTR81
            sw_p[28] = F107_OBS_LST81
            SW_PRE.append(sw_p)    
    # print((SW_OBS[0]))
    # print((SW_PRE[0]))
    SW_OBS_PRE = np.vstack((np.array(SW_OBS),np.array(SW_PRE))) 
    # inverse sort
    SW_OBS_PRE = np.flip(SW_OBS_PRE,0).astype(dtype='<U8')
    ymds = np.apply_along_axis(''.join, 1, SW_OBS_PRE[:,:3])
    SW_OBS_PRE = np.insert(SW_OBS_PRE[:,3:],0,ymds,axis=1)
    return SW_OBS_PRE 
 
def get_sw(SW_OBS_PRE,t_ymd,hour):
    '''
    Extract the necessary parameters describing the solar activity and geomagnetic activity from the space weather data.

    Usage: 
    f107A,f107,ap,aph = get_sw(SW_OBS_PRE,t_ymd,hour)

    Inputs: 
    SW_OBS_PRE -> [2d str array] Content of the space weather data
    t_ymd -> [str array or list] ['year','month','day']
    hour -> []
    
    Outputs: 
    f107A -> [float] 81-day average of F10.7 flux
    f107 -> [float] daily F10.7 flux for previous day
    ap -> [int] daily magnetic index 
    aph -> [float array] 3-hour magnetic index 

    Examples:
    >>> f107A,f107,ap,aph = get_sw(SW_OBS_PRE,t_ymd,hour)
    '''
    
    ymds = SW_OBS_PRE[:,0]
    j_, = np.where(''.join(t_ymd) == ymds)
    j = j_[0]
    f107A,f107,ap = float(SW_OBS_PRE[j,25]),float(SW_OBS_PRE[j+1,24]),int(SW_OBS_PRE[j,20])
    aph_tmp_b0 = SW_OBS_PRE[j,12:20]   
    i = int(np.floor_divide(hour,3))
    ap_c = aph_tmp_b0[i]
    aph_tmp_b1 = SW_OBS_PRE[j+1,12:20]
    aph_tmp_b2 = SW_OBS_PRE[j+2,12:20]
    aph_tmp_b3 = SW_OBS_PRE[j+3,12:20]
    aph_tmp = np.hstack((aph_tmp_b3,aph_tmp_b2,aph_tmp_b1,aph_tmp_b0))[::-1].astype(float)
    apc_index = 7-i
    aph_c369 = aph_tmp[apc_index:apc_index+4]
    aph_1233 = np.average(aph_tmp[apc_index+4:apc_index+12])
    aph_3657 = np.average(aph_tmp[apc_index+12:apc_index+20])
    aph = np.hstack((ap,aph_c369,aph_1233,aph_3657))
    return f107A,f107,ap,aph



def calc_mean_rho_total_over_orbit(time, alt, inc, f107a=150, f107=150, ap=4, met="gtd7", n=100):
    """Mean density over one orbit for a given time and solar/geomagnetic indices.

        Parameters
        ----------
        time: datetime
        UT [-].
        alt: float or array_like
            Altitude in [m].
        inc: float or array_like
            Geodetic latitude in [degrees N].
        f107a: float or array_like
            87 day average solar activity index [-].
        f107: float or array_like
            Solar activity index [-].
        ap: float or array_like
        Geomagnetic activity index [-].
        method: string
            Method used in NRLMSISE-00 model (including AO or not) [-].
        n: integer
            Number of sample points for averaging [-].

        Returns
        -------
        rho_mean: float
            Mean density [kg/m^3].
        T_mean: float 
            Mean temperature [K].
        """
    r = R_Earth.to(u.km).value + alt
    # define evenly spaced points around orbit
    ang = np.linspace(0,2*np.pi,n+1)
    ang = ang[:-1]
    coord = np.zeros((3,n))
    coord_t = np.zeros((3,n))
    coord = np.array([r*np.cos(ang),r*np.sin(ang),np.zeros(n)])

    R = rot_mat(1,inc*np.pi/180)

    lat = np.zeros(n)
    lon = np.zeros(n)
    rho = np.zeros(n)

    for i in range(n):
        coord_t[:,i] = R@coord[:,i]
        x = coord_t[0,i]
        y = coord_t[1,i]
        z = coord_t[2,i]
        lon[i] = np.arctan2(y,x)*180/np.pi
        lat[i] = 90-np.arccos(z/r)*180/np.pi

        # time = time + timedelta(hours=(lon[i]*12/180))

        # NRLMSISE-00 Model
        d = msise_flat(time, alt, lat[i], lon[i], f107a, f107, ap, method=met)
        rho[i] = d[5]

    rho_mean = np.mean(rho)
    return (rho_mean * u.g/((u.m*1e-2)**3)).to(u.kg/u.m**3), (rho * u.g/((u.m*1e-2)**3)).to(u.kg/u.m**3)


def rot_mat(ax,angle):
    """Mean density over one orbit for a given time and solar/geomagnetic indices.

        Parameters
        ----------
        ax: integer
        Axis of rotation: 1,2 or 3 [-].
    angle: float
        Angle of rotation [rad].

        Returns
        -------
        rot_mat: np.array
        DCM, rotation matrix [-].
        """
    s = np.sin(angle)
    c = np.cos(angle)

    if ax==1:
        rot_mat = np.array([[1,0,0],[0,c,-s],[0,s,c]])
    elif ax==2:
        rot_mat = np.array([[0,c,-s],[0,1,0],[0,s,c]])
    elif ax==3:
        rot_mat = np.array([[0,c,-s],[0,s,c],[0,0,1]])
    else:
        raise ValueError("Axis must be 1,2 or 3")
    return rot_mat