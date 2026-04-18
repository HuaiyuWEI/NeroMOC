"""
Prepare CMIP6 mascon predictors and MOC targets for neural-network training.

"""


import os
import scipy.io as sio
import h5py
import matplotlib.pyplot as plt
import numpy as np

from _path_utils import ensure_directory, require_existing_directory, require_existing_file
from _preprocess_utils import (
    apply_low_pass_filter,
    plot_random_clean_series as shared_plot_random_clean_series,
)


def select_realization_subset(output_dir, testdata, all_data):
    """Return the realization range and filename suffix for the chosen run mode."""

    if all_data:
        return range(1, 41), '_r1_r40'
    if testdata:
        if 'ACCESS_SSP585' in os.path.normpath(output_dir) and '2100-2300' in output_dir:
            return range(1, 6), '_r1_r5'
        return range(36, 41), '_r36_r40'
    return range(1, 36), '_r1_r35'


#%% User settings

# Example choices:
# r'E:\Data_CMIP6\ACCESS_historical'
# r'E:\Data_CMIP6\ACCESS_SSP126'
# r'E:\Data_CMIP6\ACCESS_SSP245'
# r'E:\Data_CMIP6\ACCESS_SSP370'
# r'E:\Data_CMIP6\ACCESS_SSP585'
output_dir = require_existing_directory(r'E:\Data_CMIP6\ACCESS_SSP585', 'CMIP data')
os.chdir(output_dir)


testdata = 0
all_data = 0


MOCstr = 'ASMOC'
output_subdir = os.path.join(output_dir, MOCstr)


ind_file, str_file = select_realization_subset(output_dir, testdata, all_data)



##%%% filter config
# Calculate the sampling rate (monthly data)
LPF_month = 24 # 2-year LPF
sampling_rate = 1  # Data is sampled monthly    
cutoff_freq = 1 / LPF_month
order = 5
padding_length = 2 * LPF_month



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

ensure_directory(output_subdir)

#%% concatenate ocean bottom pressure on mascons from different realizations

obp_mascon_list = []
obp_mascon_LPF_list = []
realization_index = []  # to track which time step came from which realization

for rlz in ind_file: 
    fname = f'Mascon_V5_OBP_r{rlz}.mat'
    fname = require_existing_file(fname, f'OBP mascon input for realization {rlz}')
    with h5py.File(fname, 'r') as f:
        data = np.array(f['Input_vars_mascon'])  # shape: [time, position]
        Basin_id = np.array(f['Basin_id'])  # shape: [position]
        data_lat = np.array(f['lat_mascon_center'])  # shape: [position]
        data_lon = np.array(f['lon_mascon_center'])  # shape: [position]
    
        # Ensure 1D shape
        data_lat = data_lat.squeeze()
        data_lon = data_lon.squeeze()
        
        # Mask Atlantic and Southern Ocean
        Basin_id = Basin_id.squeeze()
        mask = Basin_id == 1
        data = data[:, mask]
        data_lat = data_lat[mask]
        data_lon = data_lon[mask]

        temp = data
        plt.plot(np.nanmean(temp, axis=1))
        
        # Remove basin-mean per time step
        temp = temp - np.nanmean(temp, axis=1, keepdims=True)
        
        # Apply low-pass filter
        temp_LPF = apply_low_pass_filter(temp, cutoff_freq, order, sampling_rate, padding_length)

        # Plot random grid time series for verification
        plot_random_clean_series(temp, temp_LPF)

        # Plot
        plt.figure(figsize=(16, 6))
        scatter = plt.scatter(data_lon, data_lat, c=np.std(temp,axis=0), 
                              cmap='OrRd', s=20, edgecolor='k',
                              vmin=50, vmax=500 )
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('STD of Ocean Bottom Pressure on mascons')
        plt.colorbar(scatter, label='Data Value')
        plt.grid(True)
        plt.show()
        
        # Plot
        plt.figure(figsize=(16, 6))
        scatter = plt.scatter(data_lon, data_lat, c=np.std(temp_LPF,axis=0), 
                              cmap='OrRd', s=20, edgecolor='k',
                              vmin=50, vmax=500 )
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('STD of Ocean Bottom Pressure (LPFed) on mascons')
        plt.colorbar(scatter, label='Data Value')
        plt.grid(True)
        plt.show()


        # Append results
        obp_mascon_list.append(temp)
        obp_mascon_LPF_list.append(temp_LPF)
        realization_index.extend([rlz] * temp.shape[0])  # repeat rlz for each time step

obp_mascon_ALL = np.concatenate(obp_mascon_list, axis=0)       # [total_time, lat, lon]
obp_mascon_LPF_ALL = np.concatenate(obp_mascon_LPF_list, axis=0)
realization_index = np.array(realization_index)  # [total_time]


print(obp_mascon_ALL.shape)          
print(realization_index.shape)  


np.savez(os.path.join(output_dir,MOCstr,'obp_mascon_V5'+str_file+'.npz'),
                    obp_mascon_V5_ALL=obp_mascon_ALL,
                    obp_mascon_V5_LPF_ALL=obp_mascon_LPF_ALL,
                    mascon_lon = data_lon, 
                    mascon_lat = data_lat,
                    realization_index=realization_index)


sio.savemat(os.path.join(output_dir,MOCstr,'obp_mascon_LatLon_V5.mat'), 
            {'obp_mascon_lon': data_lon, 'obp_mascon_lat': data_lat})
                

del obp_mascon_ALL,obp_mascon_LPF_ALL




#%% concatenate SSH on mascons from different realizations

ssh_mascon_list = []
ssh_mascon_LPF_list = []
realization_index = []  # to track which time step came from which realization

for rlz in ind_file: 
    fname = f'Mascon_V5_SSH_r{rlz}.mat'
    fname = require_existing_file(fname, f'SSH mascon input for realization {rlz}')
    with h5py.File(fname, 'r') as f:
        data = np.array(f['Input_vars_mascon'])  # shape: [time, position]
        Basin_id = np.array(f['Basin_id'])  # shape: [position]
        data_lat = np.array(f['lat_mascon_center'])  # shape: [position]
        data_lon = np.array(f['lon_mascon_center'])  # shape: [position]
    
        # Ensure 1D shape
        data_lat = data_lat.squeeze()
        data_lon = data_lon.squeeze()
        
        # Mask Atlantic and Southern Ocean
        Basin_id = Basin_id.squeeze()
        mask = Basin_id == 1
        data = data[:, mask]
        data_lat = data_lat[mask]
        data_lon = data_lon[mask]
    

        temp = data
        plt.plot(np.nanmean(temp, axis=1))
        
        # Remove basin-mean per time step
        temp = temp - np.nanmean(temp, axis=1, keepdims=True)
        
        # Apply low-pass filter
        temp_LPF = apply_low_pass_filter(temp, cutoff_freq, order, sampling_rate, padding_length)

        # Plot random grid time series for verification
        plot_random_clean_series(temp, temp_LPF)

        # Plot
        plt.figure(figsize=(16, 6))
        scatter = plt.scatter(data_lon, data_lat, c=np.std(temp,axis=0), 
                              cmap='OrRd', s=20, edgecolor='k'
                               )
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('STD of SSH on mascons')
        plt.colorbar(scatter, label='Data Value')
        plt.grid(True)
        plt.show()
        

        # Append results
        ssh_mascon_list.append(temp)
        ssh_mascon_LPF_list.append(temp_LPF)
        realization_index.extend([rlz] * temp.shape[0])  # repeat rlz for each time step

ssh_mascon_ALL = np.concatenate(ssh_mascon_list, axis=0)       # [total_time, lat, lon]
ssh_mascon_LPF_ALL = np.concatenate(ssh_mascon_LPF_list, axis=0)
realization_index = np.array(realization_index)  # [total_time]


print(ssh_mascon_ALL.shape)          
print(realization_index.shape)  


np.savez(os.path.join(output_dir,MOCstr,'ssh_mascon_V5'+str_file+'.npz'),
                    ssh_mascon_V5_ALL=ssh_mascon_ALL,
                    ssh_mascon_V5_LPF_ALL=ssh_mascon_LPF_ALL,
                    mascon_lon = data_lon, 
                    mascon_lat = data_lat,
                    realization_index=realization_index)



del ssh_mascon_ALL,ssh_mascon_LPF_ALL





#%% concatenate UAS (eastward 10-m wind) on mascons from different realizations

uas_mascon_list = []
uas_mascon_LPF_list = []
realization_index = []  # to track which time step came from which realization

for rlz in ind_file: 
    fname = f'Mascon_V5_uas_r{rlz}.mat'
    fname = require_existing_file(fname, f'UAS mascon input for realization {rlz}')
    with h5py.File(fname, 'r') as f:
        data = np.array(f['Input_vars_mascon'])  # shape: [time, position]
        Basin_id = np.array(f['Basin_id'])  # shape: [position]
        data_lat = np.array(f['lat_mascon_center'])  # shape: [position]
        data_lon = np.array(f['lon_mascon_center'])  # shape: [position]
    
        # Ensure 1D shape
        data_lat = data_lat.squeeze()
        data_lon = data_lon.squeeze()
        
        # Mask Atlantic and Southern Ocean
        Basin_id = Basin_id.squeeze()
        mask = Basin_id == 1
        data = data[:, mask]
        data_lat = data_lat[mask]
        data_lon = data_lon[mask]
    

        temp = data
        plt.plot(np.nanmean(temp, axis=1))
        # Remove basin-mean per time step
        # temp = temp - np.nanmean(temp, axis=1, keepdims=True)
        
        # Apply low-pass filter
        temp_LPF = apply_low_pass_filter(temp, cutoff_freq, order, sampling_rate, padding_length)

        # Plot random grid time series for verification
        plot_random_clean_series(temp, temp_LPF)

        # Plot
        plt.figure(figsize=(18, 6))
        scatter = plt.scatter(data_lon, data_lat, c=np.std(temp,axis=0), 
                              cmap='OrRd', s=60, edgecolor='k'
                               )
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('STD of uas on mascons')
        plt.colorbar(scatter, label='Data Value')
        plt.grid(True)
        plt.show()
        
        # Plot
        plt.figure(figsize=(18, 6))
        scatter = plt.scatter(data_lon, data_lat, c=np.std(temp_LPF[1828:,:],axis=0), 
                              cmap='OrRd', s=60, edgecolor='k',
                              vmin=0, vmax=1)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('STD of uas (LPFed) on mascons (2002-2015)')
        plt.colorbar(scatter, label='Data Value')
        plt.grid(True)
        plt.show()

        
        
        # Append results
        uas_mascon_list.append(temp)
        uas_mascon_LPF_list.append(temp_LPF)
        realization_index.extend([rlz] * temp.shape[0])  # repeat rlz for each time step

uas_mascon_ALL = np.concatenate(uas_mascon_list, axis=0)       # [total_time, lat, lon]
uas_mascon_LPF_ALL = np.concatenate(uas_mascon_LPF_list, axis=0)
realization_index = np.array(realization_index)  # [total_time]


print(uas_mascon_ALL.shape)          
print(realization_index.shape)  


np.savez(os.path.join(output_dir,MOCstr,'uas_mascon_V5'+str_file+'.npz'),
                    uas_mascon_V5_ALL=uas_mascon_ALL,
                    uas_mascon_V5_LPF_ALL=uas_mascon_LPF_ALL,
                    mascon_lon = data_lon, 
                    mascon_lat = data_lat,
                    realization_index=realization_index)



del uas_mascon_ALL,uas_mascon_LPF_ALL
#%% MOC

    
MOC_list = []
MOC_LPF_list = []
realization_index = []  # to track which time step came from which realization


for rlz in ind_file: 
    fname = f'FullDepth_ASMOC_interp_gr_r{rlz}.mat'
    fname = require_existing_file(fname, f'MOC target input for realization {rlz}')
    with h5py.File(fname, 'r') as f:
        data = np.array(f['Psi_ASMOC_interp'])  # [var, time, lat, lon]
        Lev_psi = np.array(f['RHO_ASMOC_interp'])
        Lat_psi = np.array(f['LAT_ASMOC_interp'])
        temp = np.transpose(data, (0, 2, 1))
        rho2_full = Lev_psi[:,0]
        lat_psi = Lat_psi[0,:]
        
        temp = temp/1e6;
        
        
        
        # Apply low-pass filter
        temp_LPF = apply_low_pass_filter(temp, cutoff_freq, order, sampling_rate, padding_length)
        # Plot random grid time series for verification
        plot_random_clean_series(temp, temp_LPF)

        # Append results
        MOC_list.append(temp)
        MOC_LPF_list.append(temp_LPF)
        realization_index.extend([rlz] * temp.shape[0])  # repeat rlz for each time step


MOC_ALL = np.concatenate(MOC_list, axis=0)       # [total_time, lat, lon]
MOC_LPF_ALL = np.concatenate(MOC_LPF_list, axis=0)
realization_index = np.array(realization_index)  # [total_time]


print(MOC_ALL.shape)          
print(realization_index.shape)  


np.savez(os.path.join(output_dir,MOCstr,'MOC'+str_file+'.npz'),
                    MOC_ALL=MOC_ALL,
                    MOC_LPF_ALL=MOC_LPF_ALL,
                    rho2_full=rho2_full,
                    lat_psi = lat_psi,
                    realization_index=realization_index)


from scipy.io import savemat

# Save as .mat
savemat(os.path.join(output_dir, MOCstr, 'MOC'+str_file+'.mat'),
        {'MOC_ALL': MOC_ALL,
         'MOC_LPF_ALL': MOC_LPF_ALL,
         'rho2_full': rho2_full,
         'lat_psi': lat_psi,
         'realization_index': realization_index})



















