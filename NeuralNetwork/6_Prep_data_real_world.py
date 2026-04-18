"""
Pre-process real-world satellite data for the Neural Network
Pre-process AMOC observations

"""

#%%


import os
import scipy.io as sio
import h5py
import matplotlib.pyplot as plt
import numpy as np

from _path_utils import ensure_directory, require_existing_file
from _preprocess_utils import (
    apply_low_pass_filter,
    plot_random_clean_series as shared_plot_random_clean_series,
)
OBS_OUTPUT_ROOT = r'D:\OneDrive - University of California\PythonWork\mascon'
GRACE_INPUT_FILE = r'D:\OneDrive - University of California\PythonWork\mascon\GRACE_OBP_200204_202505_V5.mat'
DUACS_INPUT_FILE = r'E:\Data_SSH\DUACS\daily\DUACS_mascon_200204_202504_V5.mat'
CCMP_INPUT_FILE = r'E:\Data_CCMPWind\CCMPWind_mascon_200204_202504_V5.mat'
ERA5_INPUT_FILE = r'E:\Data_ERA5\ERA5_mascon_200201_202507_V5.mat'
MOCid = 1
MOCstr = 'ASMOC'
mascon_output_dir = os.path.join(OBS_OUTPUT_ROOT, MOCstr)
ensure_directory(mascon_output_dir)


#%% filter config
# Calculate the sampling rate (monthly data)
LPF_month = 24 # 2-year LPF
sampling_rate = 1  # Data is sampled monthly    
cutoff_freq = 1 / LPF_month
order = 5
padding_length = 24 #2 * LPF_month

def plot_random_clean_series(temp, temp_LPF, max_attempts=100):
    """
    Plots a time series at a random clean (non-NaN) location in temp.
    
    Supports both [time, lat, lon] and [time, position] shapes.

    Parameters:
        temp        : np.ndarray, shape [time, lat, lon] or [time, position] — original data
        temp_LPF    : np.ndarray, same shape as `temp` — low-pass filtered data
        max_attempts: int — number of random attempts to find a clean time series
    """
    return shared_plot_random_clean_series(temp, temp_LPF, max_attempts)

#%% Grace data
fname = require_existing_file(GRACE_INPUT_FILE, 'GRACE mascon input')
with h5py.File(fname, 'r') as f:
    # from 17-Apr-2002 to 16-May-2025
    data = np.array(f['obp_grace_monthly']).T  # [time, lat, lon]; pa
    Basin_id = np.array(f['Basin_id'])  # shape: [position]
    data_lat = np.array(f['lat_mascon_center'])  # shape: [position]
    data_lon = np.array(f['lon_mascon_center'])  # shape: [position]
    GAD = np.array(f['GAD_grace_monthly']).T


# Ensure 1D shape
print(data.shape)
data_lat = data_lat.squeeze()
data_lon = data_lon.squeeze()


# Mask Atlantic 

Basin_id = Basin_id.squeeze()
mask = Basin_id == 1
data = data[:, mask]
data_lat = data_lat[mask]
data_lon = data_lon[mask]

GAD= GAD[:, mask]


temp = data
temp2 = GAD
# Plot

plt.figure(figsize=(14, 4))
scatter = plt.scatter(data_lon, data_lat, c=np.mean(temp,axis=0), cmap='RdBu', s=30, edgecolor='k')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Mean GRACE Ocean Bottom Pressure 2002-2025')
plt.colorbar(scatter, label='Data Value')
plt.grid(True)
plt.show()
        
        
# Remove basin-mean per time step
temp = temp - np.nanmean(temp, axis=1, keepdims=True)

temp2 = temp2 - np.nanmean(temp2, axis=1, keepdims=True)


# Apply low-pass filter
temp_LPF = apply_low_pass_filter(temp, cutoff_freq, order, sampling_rate, padding_length)

temp2_LPF = apply_low_pass_filter(temp2, cutoff_freq, order, sampling_rate, padding_length)


# Plot random grid time series for verification
plot_random_clean_series(temp, temp_LPF)


# Plot random grid time series for verification
plot_random_clean_series(temp2, temp2_LPF)




# Plot
plt.figure(figsize=(18, 6))
scatter = plt.scatter(data_lon, data_lat, c=np.std(temp,axis=0), 
                      cmap='OrRd', s=60, edgecolor='k',
                      vmin=20, vmax=200 )
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('STD of Ocean Bottom Pressure on mascons')
plt.colorbar(scatter, label='Data Value')
plt.grid(True)
plt.show()

# Plot
plt.figure(figsize=(18, 6))
scatter = plt.scatter(data_lon, data_lat, c=np.std(temp_LPF[:154,:],axis=0), 
                      cmap='OrRd', s=60, edgecolor='k',
                      vmin=20, vmax=200 )
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('STD of Ocean Bottom Pressure (LPFed) on mascons (2002-2015)')
plt.colorbar(scatter, label='Data Value')
plt.grid(True)
plt.show()



np.savez(os.path.join(mascon_output_dir,'obp_GRACE.npz'),
                    obp_GRACE=temp,
                    obp_GRACE_LPF_ALL=temp_LPF,
                    GAD_GRACE=temp2,
                    GAD_GRACE_LPF_ALL=temp2_LPF,
                    obp_GRACE_lon = data_lon, 
                    obp_GRACE_lat = data_lat)




del temp,temp_LPF



#%% DUACS SSH data

fname = require_existing_file(DUACS_INPUT_FILE, 'DUACS mascon input')
with h5py.File(fname, 'r') as f:
    # from 1-Apr-2002 to 1-Oct-2024
    data = np.array(f['ssh_mascon_NoNan'])  # [time, lat, lon]; pa
    Basin_id = np.array(f['Basin_id'])  # shape: [position]
    data_lat = np.array(f['lat_mascon_center'])  # shape: [position]
    data_lon = np.array(f['lon_mascon_center'])  # shape: [position]

# Ensure 1D shape
print(data.shape)
data_lat = data_lat.squeeze()
data_lon = data_lon.squeeze()


# Mask Atlantic 

Basin_id = Basin_id.squeeze()
mask = Basin_id == 1
data = data[:, mask]
data_lat = data_lat[mask]
data_lon = data_lon[mask]



        
temp = data
# Remove basin-mean per time step
plt.plot(np.nanmean(temp, axis=1, keepdims=True))
temp = temp - np.nanmean(temp, axis=1, keepdims=True)


# Apply low-pass filter
temp_LPF = apply_low_pass_filter(temp, cutoff_freq, order, sampling_rate, padding_length)

# Plot random grid time series for verification
plot_random_clean_series(temp, temp_LPF)



# Plot
plt.figure(figsize=(18, 6))
scatter = plt.scatter(data_lon, data_lat, c=np.std(temp,axis=0), 
                      cmap='OrRd', s=60, edgecolor='k',
                      vmin=0, vmax=0.15 )
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('STD of SSH on mascons')
plt.colorbar(scatter, label='Data Value')
plt.grid(True)
plt.show()

# Plot
plt.figure(figsize=(18, 6))
scatter = plt.scatter(data_lon, data_lat, c=np.std(temp_LPF[:154,:],axis=0), 
                      cmap='OrRd', s=60, edgecolor='k',
                      vmin=0, vmax=0.06 )
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('STD of SSH (LPFed) on mascons (2002-2015)')
plt.colorbar(scatter, label='Data Value')
plt.grid(True)
plt.show()



# temp_LPF[np.isnan(temp_LPF)] = 0
# temp[np.isnan(temp)] = 0



np.savez(os.path.join(mascon_output_dir,'ssh_DUACS.npz'),
                    ssh_DUACS=temp,
                    ssh_DUACS_LPF_ALL=temp_LPF,
                    ssh_DUACS_lon = data_lon, 
                    ssh_DUACS_lat = data_lat)




del temp,temp_LPF

#%% CCMP zonal wind speed data

fname = require_existing_file(CCMP_INPUT_FILE, 'CCMP wind mascon input')

with h5py.File(fname, 'r') as f:
    # from Jan-2002 to Dec-2024
    data = np.array(f['u10_mascon'])  # [time, lat, lon]; pa
    Basin_id = np.array(f['Basin_id'])  # shape: [position]
    data_lat = np.array(f['lat_mascon_center'])  # shape: [position]
    data_lon = np.array(f['lon_mascon_center'])  # shape: [position]

# Ensure 1D shape
print(data.shape)
data_lat = data_lat.squeeze()
data_lon = data_lon.squeeze()


# Mask Atlantic 

Basin_id = Basin_id.squeeze()
mask = Basin_id == 1
data = data[:, mask]
data_lat = data_lat[mask]
data_lon = data_lon[mask]



        
temp = data
# Remove basin-mean per time step
# temp = temp - np.nanmean(temp, axis=1, keepdims=True)


# Apply low-pass filter
temp_LPF = apply_low_pass_filter(temp, cutoff_freq, order, sampling_rate, padding_length)

# Plot random grid time series for verification
plot_random_clean_series(temp, temp_LPF)



# Plot
plt.figure(figsize=(18, 6))
scatter = plt.scatter(data_lon, data_lat, c=np.std(temp,axis=0), 
                      cmap='OrRd', s=60, edgecolor='k',
                      vmin=0, vmax=10 )
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('STD of uas on mascons')
plt.colorbar(scatter, label='Data Value')
plt.grid(True)
plt.show()

# Plot
plt.figure(figsize=(18, 6))
scatter = plt.scatter(data_lon, data_lat, c=np.std(temp_LPF[:154,:],axis=0), 
                      cmap='OrRd', s=60, edgecolor='k',
                      vmin=0, vmax=5 )
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('STD of uas (LPFed) on mascons (2002-2015)')
plt.colorbar(scatter, label='Data Value')
plt.grid(True)
plt.show()


# Plot
plt.figure(figsize=(18, 6))
scatter = plt.scatter(data_lon, data_lat, c=np.mean(temp_LPF[:154,:],axis=0), 
                      cmap='OrRd', s=60, edgecolor='k',
                      vmin=0, vmax=5 )
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Mean of uas (LPFed) on mascons (2002-2015)')
plt.colorbar(scatter, label='Data Value')
plt.grid(True)
plt.show()


temp_LPF[np.isnan(temp_LPF)] = 0
temp[np.isnan(temp)] = 0

np.savez(os.path.join(mascon_output_dir,'uas_CCMP.npz'),
                    uas_CCMP=temp,
                    uas_CCMP_LPF_ALL=temp_LPF,
                    uas_CCMP_lon = data_lon, 
                    uas_CCMP_lat = data_lat)




del temp,temp_LPF



#%% RAPID data
import xarray as xr

# Load the NetCDF file
ds = xr.open_dataset('E:\Data_RAPID\moc_transports.nc')

# See the structure of the dataset
print(ds)

moc_strength = ds['moc_mar_hc10'].values
moc_time = ds['time'].values


monthly_mean = ds['moc_mar_hc10'].resample(time='1M').mean()
print(monthly_mean)


RAPID_monthly = monthly_mean.values

# Apply low-pass filter
RAPID_monthly_LPF = apply_low_pass_filter(RAPID_monthly, cutoff_freq, order, sampling_rate, padding_length)

t0_year = 2004 + 4/12
t_year = t0_year + np.arange(RAPID_monthly.shape[0])/12
# Plot the valid time series
plt.figure(figsize=(10, 4))
plt.plot(t_year,RAPID_monthly, label='Original', linestyle='--', color='gray')
plt.plot(t_year,RAPID_monthly_LPF, label='Low-pass filtered', color='blue')
plt.xlabel('Year')
plt.ylabel('AMOC observed by RAPID')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


sio.savemat(os.path.join('E:\Data_RAPID', 'Rapid_LPF.mat'), {
    'RAPID_monthly': RAPID_monthly,
    'RAPID_monthly_LPF': RAPID_monthly_LPF,
    't_year':t_year
})


#%% RAPID data Full depth
import xarray as xr

# Load the NetCDF file
ds = xr.open_dataset('E:\Data_RAPID\meridional_transports.nc')

# See the structure of the dataset
print(ds)

stream_depth = ds['stream_depth'].resample(time='1M').mean()
stream_sigma2 = ds['stream_sigma2'].resample(time='1M').mean()
stream_sigma0 = ds['stream_sigma0'].resample(time='1M').mean()
moc_time = ds['time'].values

depth = ds['depth'].values
sigma2 = ds['sigma2'].values
sigma0 = ds['sigma0'].values
# Apply low-pass filter
stream_depth_LPF = apply_low_pass_filter(stream_depth, cutoff_freq, order, sampling_rate, padding_length)
stream_sigma2_LPF = apply_low_pass_filter(stream_sigma2, cutoff_freq, order, sampling_rate, padding_length)
stream_sigma0_LPF = apply_low_pass_filter(stream_sigma0, cutoff_freq, order, sampling_rate, padding_length)


t0_year = 2004 + 4/12
t_year = t0_year + np.arange(stream_depth.shape[0])/12


sio.savemat(os.path.join('E:\Data_RAPID', 'Rapid_FullDepth_LPF.mat'), {
    'stream_depth_LPF': stream_depth_LPF,
    'stream_sigma2_LPF': stream_sigma2_LPF,
    'stream_sigma0_LPF': stream_sigma0_LPF,
    'sigma2':sigma2,
    'sigma0':sigma0,
    'depth':depth,
    't_year':t_year
})


#%% OSNAP data
import xarray as xr

# Load the NetCDF file
ds = xr.open_dataset('E:\Data_OSNAP\OSNAP_Streamfunction_201408_202207_2025.nc')

# See the structure of the dataset
print(ds)

moc_strength = ds['T_ALL'].values.max(axis=0)
moc_time = ds['TIME'].values


monthly_mean = moc_strength
plt.plot(moc_time,monthly_mean)


OSNAP_monthly = monthly_mean

# Apply low-pass filter
OSNAP_monthly_LPF = apply_low_pass_filter(OSNAP_monthly, cutoff_freq, order, sampling_rate, padding_length)

t0_year = 2014 + 8/12
t_year = t0_year + np.arange(OSNAP_monthly.shape[0])/12
# Plot the valid time series
plt.figure(figsize=(10, 4))
plt.plot(t_year,OSNAP_monthly, label='Original', linestyle='--', color='gray')
plt.plot(t_year,OSNAP_monthly_LPF, label='Low-pass filtered', color='blue')
plt.xlabel('Year')
plt.ylabel('AMOC observed by OSNAP')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


sio.savemat(os.path.join('E:\Data_OSNAP', 'OSNAP_LPF.mat'), {
    'OSNAP_monthly': OSNAP_monthly,
    'OSNAP_monthly_LPF': OSNAP_monthly_LPF,
    't_year':t_year
})
#%% MOVE data
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

path = r'E:\Data_MOVE\OS_MOVE_20000206-20221014_DPR_VOLUMETRANSPORT.nc'
ds = xr.open_dataset(path,decode_times=False)

moc_strength = ds['TRANSPORT_TOTAL'].values
moc_time = ds['TIME'].values

plt.plot(moc_strength)
plt.plot(moc_time)

valid = moc_strength<1e36

plt.plot(moc_strength[valid])
plt.plot(moc_time[valid])

# Convert "days since 1950-01-01" → pandas datetime
epoch = pd.Timestamp('1950-01-01T00:00:00')
dt = epoch + pd.to_timedelta(moc_time[valid], unit='D')

# Monthly mean (labelled at month start); use 'M' for month-end labels if preferred
s = pd.Series(moc_strength[valid], index=dt).sort_index()

moc_monthly = (s.resample('1M').mean()
                 .asfreq('1M')
                 .interpolate(method='time'))


# Plot
plt.figure()
plt.plot(moc_monthly.index, moc_monthly.values)
plt.xlabel('Time')
plt.ylabel('Monthly mean TRANSPORT_TOTAL')
plt.title('Monthly MOC strength')
plt.grid(True)


MOVE_monthly = moc_monthly.values

# Apply low-pass filter
MOVE_monthly_LPF = apply_low_pass_filter(MOVE_monthly, cutoff_freq, order, sampling_rate, padding_length)


t_year = moc_monthly.index
# Plot the valid time series
plt.figure(figsize=(10, 4))
plt.plot(t_year,MOVE_monthly, label='Original', linestyle='--', color='gray')
plt.plot(t_year,MOVE_monthly_LPF, label='Low-pass filtered', color='blue')
plt.xlabel('Year')
plt.ylabel('AMOC observed by MOVE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


sio.savemat(os.path.join('E:\Data_MOVE', 'MOVE_LPF.mat'), {
    'MOVE_monthly': MOVE_monthly,
    'MOVE_monthly_LPF': MOVE_monthly_LPF,
    't_year':t_year
})




#%% ECCO data V4r3
fname = 'E:\Data_ECCO\ECCOV4r3\myproducts_monthly\PSItot_AtlOnly.mat';
data = sio.loadmat(fname)
MOC_ecco_mon = data['PSItot'].T/1e6

MOC_ecco_LPF = apply_low_pass_filter(MOC_ecco_mon, cutoff_freq, order, sampling_rate, padding_length)


t0_year = 1992 + 1/12
t_year = t0_year + np.arange(MOC_ecco_mon.shape[0])/12


sio.savemat(os.path.join('E:\Data_ECCO\ECCOV4r3', 'PSI_LPF.mat'), {
    'MOC_ecco_LPF': MOC_ecco_LPF,
    'MOC_ecco': MOC_ecco_mon,
    't_year_ecco':t_year,
})




#%% ECCO data V4r4
import pandas as pd

fname = 'E:\Data_ECCO\ECCOV4r4\myproducts_daily\PSItot.mat';
data = sio.loadmat(fname)
MOC_ecco = data['PSI'].T


# daily time axis
nt, nz, ny = MOC_ecco.shape
time_daily = pd.date_range('1992-01-01', periods=nt, freq='D')

# wrap in an xarray DataArray
da = xr.DataArray(
    MOC_ecco,
    coords={'time': time_daily,
            'z': np.arange(nz),
            'y': np.arange(ny)},
    dims=('time', 'z', 'y'),
    name='MOC_ecco'
)

# monthly mean (labelled by month-end; use 'MS' if you prefer month-start)
MOC_ecco_mon = da.resample(time='M').mean('time').values/1e6


# Apply low-pass filter
MOC_ecco_LPF = apply_low_pass_filter(MOC_ecco_mon, cutoff_freq, order, sampling_rate, padding_length)


plot_random_clean_series(MOC_ecco_mon, MOC_ecco_LPF)


t0_year = 1992 + 1/12
t_year = t0_year + np.arange(MOC_ecco_mon.shape[0])/12


sio.savemat(os.path.join('E:\Data_ECCO\ECCOV4r4', 'PSI_LPF.mat'), {
    'MOC_ecco_LPF': MOC_ecco_LPF,
    'MOC_ecco': MOC_ecco_mon,
    't_year_ecco':t_year,
})





#%% SCOTIA

import h5py

fname = r'D:\OneDrive - University of California\MATLAB Codes\MOC\SCOTIA_timeseries.mat'

with h5py.File(fname, 'r') as f:
    print(list(f.keys()))
    time = f['t'][:].T
    MOC_SCOTIA = f['MOC'][:]


# Apply low-pass filter
MOC_SCOTIA_LPF = apply_low_pass_filter(MOC_SCOTIA, cutoff_freq, order, sampling_rate, padding_length)


t0_year = 2004 + 1/12
t_year = t0_year + np.arange(time.shape[0])/12
# Plot the valid time series
plt.figure(figsize=(10, 4))
plt.plot(t_year.reshape(-1,1),MOC_SCOTIA, label='Original', linestyle='--', color='gray')
plt.plot(t_year.reshape(-1,1),MOC_SCOTIA_LPF, label='Low-pass filtered', color='blue')
plt.xlabel('Year')
plt.ylabel('AMOC via SCOTIA')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



sio.savemat(os.path.join('D:\OneDrive - University of California\MATLAB Codes\MOC', 'SCOTIA_LPF.mat'), {
    'MOC_SCOTIA_LPF': MOC_SCOTIA_LPF,
    'MOC_SCOTIA': MOC_SCOTIA,
    't_year_SCOTIA':t_year,
})



#%% ERA5 zonal wind speed data (not used)

fname = require_existing_file(ERA5_INPUT_FILE, 'ERA5 mascon input')

with h5py.File(fname, 'r') as f:
    # from 1-Jan-2002 to 1-July-2025
    data = np.array(f['u10_mascon'])  # [time, lat, lon]; pa
    Basin_id = np.array(f['Basin_id'])  # shape: [position]
    data_lat = np.array(f['lat_mascon_center'])  # shape: [position]
    data_lon = np.array(f['lon_mascon_center'])  # shape: [position]

# Ensure 1D shape
print(data.shape)
data_lat = data_lat.squeeze()
data_lon = data_lon.squeeze()


# Mask Atlantic 

Basin_id = Basin_id.squeeze()
mask = Basin_id == 1
data = data[:, mask]
data_lat = data_lat[mask]
data_lon = data_lon[mask]



        
temp = data
# Remove basin-mean per time step
# temp = temp - np.nanmean(temp, axis=1, keepdims=True)


# Apply low-pass filter
temp_LPF = apply_low_pass_filter(temp, cutoff_freq, order, sampling_rate, padding_length)

# Plot random grid time series for verification
plot_random_clean_series(temp, temp_LPF)



# Plot
plt.figure(figsize=(18, 6))
scatter = plt.scatter(data_lon, data_lat, c=np.std(temp,axis=0), 
                      cmap='OrRd', s=60, edgecolor='k',
                      vmin=0, vmax=5 )
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('STD of uas on mascons')
plt.colorbar(scatter, label='Data Value')
plt.grid(True)
plt.show()

# Plot
plt.figure(figsize=(18, 6))
scatter = plt.scatter(data_lon, data_lat, c=np.std(temp_LPF[:154,:],axis=0), 
                      cmap='OrRd', s=60, edgecolor='k',
                      vmin=0, vmax=2 )
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('STD of uas (LPFed) on mascons (2002-2015)')
plt.colorbar(scatter, label='Data Value')
plt.grid(True)
plt.show()


# Plot
plt.figure(figsize=(18, 6))
scatter = plt.scatter(data_lon, data_lat, c=np.mean(temp_LPF[:154,:],axis=0), 
                      cmap='RdBu_r', s=60, edgecolor='k',
                      vmin=-10, vmax=10 )
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Mean of uas (LPFed) on mascons (2002-2015)')
plt.colorbar(scatter, label='Data Value')
plt.grid(True)
plt.show()


temp_LPF[np.isnan(temp_LPF)] = 0
temp[np.isnan(temp)] = 0

np.savez(os.path.join(mascon_output_dir,'uas_ERA5.npz'),
                    uas_ERA5=temp,
                    uas_ERA5_LPF_ALL=temp_LPF,
                    uas_ERA5_lon = data_lon, 
                    uas_ERA5_lat = data_lat)




del temp,temp_LPF






