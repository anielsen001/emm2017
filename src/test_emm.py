import numpy as np
import xarray as xr
import ctypes as ct
import pandas as pd

df_test = pd.read_table('EMM2017TestValues.txt',
                        skiprows=18,
                        header=None,
                        delim_whitespace=True,
                      names=['date','alt_km','lat_deg','lon_deg','dec_deg','inc_deg','H','x','y','z','F','dD/dt','dI/dt','dH/dt','dx/dt','dy/dt','dz/dt','df/dt'])

libemm = ct.cdll.LoadLibrary('./emm_point_sub.so')

x = ct.c_double()
y = ct.c_double()
z = ct.c_double()
T = ct.c_double()
D = ct.c_double()
mI = ct.c_double()

yeardec = 2000.0
alt_km = 0.0
glat = 78.3
glon = 123.7

ret = libemm.emmsub(
    ct.c_double(glat),
    ct.c_double(glon),
    ct.c_double(alt_km),
    ct.c_double(yeardec),
    ct.byref(x),
    ct.byref(y),
    ct.byref(z),
    ct.byref(T),
    ct.byref(D),
    ct.byref(mI) )

def emmsub(glat,
           glon,
           alt_km,
           yeardec):
    x = ct.c_double()
    y = ct.c_double()
    z = ct.c_double()
    T = ct.c_double()
    D = ct.c_double()
    mI = ct.c_double()

    ret = libemm.emmsub(
        ct.c_double(glat),
        ct.c_double(glon),
        ct.c_double(alt_km),
        ct.c_double(yeardec),
        ct.byref(x),
        ct.byref(y),
        ct.byref(z),
        ct.byref(T),
        ct.byref(D),
        ct.byref(mI) )

    return np.array([x.value,y.value,z.value])

xyz = [ emmsub(glat,glon,alt_km,yeardec) for glat,glon,alt_km,yeardec in zip(df_test['lat_deg'],
                                                 df_test['lon_deg'],
                                                 df_test['alt_km'],
                                                 df_test['date'])]

# this will verify that all the test cases match
assert np.allclose( np.array(xyz), df_test[['x','y','z']].to_numpy(), atol=0.1 ) 
