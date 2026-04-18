"""
Evaluate trained AMOC neural networks on out-of-sample CMIP6 experiments.

"""

import gc
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import tensorflow as tf
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from _keras_utils import load_model_with_noise_support
from _path_utils import require_existing_directory, require_existing_file
from _runtime_utils import configure_tensorflow_runtime, prepare_covariate_config



configure_tensorflow_runtime()

CMIP_DATA_ROOT = r'E:\Data_CMIP6'
MODEL_BASE_ROOT = r'E:\Analysis2026\ACCESS_hist+SSP585'
CMIP_DATA_ROOT = require_existing_directory(CMIP_DATA_ROOT, 'CMIP data root')
MODEL_BASE_ROOT = require_existing_directory(MODEL_BASE_ROOT, 'Model output root')


# ========== User settings ==========
MOC_str = 'ASMOC'
covariate_names = "obp_mascon_V5,ssh_mascon_V5,uas_mascon_V5"
AMOC26_input = 0
AMOC56_input = 0

run_config = prepare_covariate_config(covariate_names, AMOC26_input, AMOC56_input)
covariate_names = run_config.covariate_names
inputVar = run_config.input_var
if run_config.noise_size:
    print("NoiseSize =", run_config.noise_size)

NNpath_base = MODEL_BASE_ROOT

#%% filter config

from scipy.signal import butter, sosfiltfilt
def apply_low_pass_filter(data, cutoff_freq, order=5, sampling_rate=1, padding_length=None):
    """
    Apply a Butterworth low-pass filter to each time series.

    Expected shapes:
    - 2D: [time, feature]
    - 3D: [ensemble, feature, time]
    """
    sos = butter(order, cutoff_freq, btype='low', output='sos', analog=False, fs=sampling_rate)

    if padding_length is None:
        padding_length = 0

    # Determine dimensions
    ndim = data.ndim
    if ndim == 2:
        nt, ny = data.shape
        filtered_data = np.empty_like(data)

        for j in range(ny):
            ts = data[:, j]

            if np.any(np.isnan(ts)):
                filtered_data[:, j] = ts  # leave as-is
                continue

            padded = np.pad(ts, (padding_length, padding_length), mode='reflect')
            filtered = sosfiltfilt(sos, padded)
            filtered_data[:, j] = filtered[padding_length:-padding_length]

    elif ndim == 3:
        ne, nt, ny = data.shape
        filtered_data = np.empty_like(data)

        for j in range(ny):
            for i in range(ne):
                ts = data[i, j, :]

                if np.any(np.isnan(ts)):
                    filtered_data[i, j, :] = ts
                    continue

                padded = np.pad(ts, (padding_length, padding_length), mode='reflect')
                filtered = sosfiltfilt(sos, padded)
                filtered_data[i, j, :] = filtered[padding_length:-padding_length]
    else:
        raise ValueError("Input data must be 2D or 3D with time as the first dimension.")

    return filtered_data


#%%
# ========== Helper Functions ==========

def load_data(data_dir, str_file, covariate_names, LPF_data_str,AMOC26_input,AMOC56_input):
    """Load input variables and MOC strength."""
    
    data_MOC = np.load(os.path.join(data_dir, 'MOC' + str_file + '.npz'))
    rho2 = data_MOC['rho2_full']
    lat_psi=data_MOC['lat_psi']
    Nlats_psi = lat_psi.shape[0]
    Psi = data_MOC['MOC'+LPF_data_str] 
    Psi = np.transpose(Psi, (0, 2, 1)) # Time, Lev, Lat
    Nsamps = Psi.shape[0]
    Nlevs = Psi.shape[1]
        
        
    def find_nearest_lat_index(lat_array, target_lat):
        """Return the index of the latitude value in `lat_array` closest to `target_lat`."""
        lat_array = np.asarray(lat_array)
        return np.argmin(np.abs(lat_array - target_lat))
    if AMOC26_input:
        ind_y = find_nearest_lat_index(lat_psi, 26.5)
        Psi_26 = Psi[:,:,ind_y]
    if AMOC56_input:
        ind_y = find_nearest_lat_index(lat_psi, 56.5)
        Psi_56 = Psi[:,:,ind_y]

    Psi = Psi.reshape(Nsamps, Nlevs * lat_psi.shape[0])
    Psi_mask = ~np.isnan(Psi).any(axis=0)
    Psi = Psi[:, Psi_mask]
    del data_MOC
    
    InputALL = np.empty((Nsamps,0))
    InputNumInd = np.empty((0))
    
    # load input variables
    if covariate_names:
        for name in covariate_names.split(','):
            predictor_file = require_existing_file(
                os.path.join(data_dir, name + str_file + '.npz'),
                f'Predictor file for {name}',
            )
            with np.load(predictor_file) as predictor_data:
                temp = predictor_data[name + LPF_data_str]
                mascon_lon = predictor_data['mascon_lon']
                mascon_lat = predictor_data['mascon_lat']
            InputALL = np.concatenate((InputALL, temp), axis=1)
            InputNumInd = np.append(InputNumInd, temp.shape[1])
    
    if AMOC26_input:
        InputALL = np.concatenate((InputALL,Psi_26),axis = 1)
        InputNumInd = np.append(InputNumInd,Psi_26.shape[1])

    if AMOC56_input:
        InputALL = np.concatenate((InputALL,Psi_56),axis = 1)
        InputNumInd = np.append(InputNumInd,Psi_56.shape[1])
        
    return lat_psi, rho2, Psi, Psi_mask, InputALL, InputNumInd, Nsamps, Nlevs

def load_model_and_predict(NNpath, InputALL, usePCA, Psi_mask, n_folds, n_ensembles):
    """Load models and predict."""
    Nsamps = InputALL.shape[0]
    n_outputs = Psi_mask.sum()
    pred_ALL = np.empty((n_folds, Nsamps, n_outputs))

    for fold_no in range(1, n_folds+1):
        scaler_x = joblib.load(os.path.join(NNpath, f'scaler_x_fold{fold_no}.pkl'))
        scaler_y = joblib.load(os.path.join(NNpath, f'scaler_y_fold{fold_no}.pkl'))
        if usePCA:
            pca_y = joblib.load(os.path.join(NNpath, f'pca_y_fold{fold_no}.pkl'))

        X = scaler_x.transform(InputALL)
        preds_ensemble = np.empty((n_ensembles, Nsamps, n_outputs))

        for ens_no in range(1, n_ensembles+1):
            model = load_model_with_noise_support(os.path.join(NNpath, f'model_fold{fold_no}_ens{ens_no}.h5'))
            y_pred = model.predict(X, verbose=0)
            if usePCA:
                y_pred = pca_y.inverse_transform(y_pred)
            preds_ensemble[ens_no-1] = y_pred

        pred_mean = preds_ensemble.mean(axis=0)
        pred_ALL[fold_no-1] = scaler_y.inverse_transform(pred_mean)
        
        # optional cleanup after each fold
        del preds_ensemble, X, scaler_x, scaler_y
        if usePCA:
            del pca_y
        gc.collect()
    pred_mean_over_folds = np.mean(pred_ALL, axis=0)
    return pred_mean_over_folds

def compute_r2_and_corr(Psi, pred_mean, Psi_mask, Nlevs, Nlats, n_realizations,
                        cutoff_freq=1/120, order=5, sampling_rate=12, padding_length=60):
    """Apply 10-year LPF to the prediction and target, 
    then compute R² and Pearson correlation."""

    n_realizations = int(n_realizations)
    samples_per_realization = Psi.shape[0] // n_realizations
    n_outputs = Psi.shape[1]
    
    Psi_LPFafter = np.empty((0, n_outputs))
    pred_mean_LPFafter = np.empty((0, n_outputs))
    
    # Low-pass filter
    for i in range(n_realizations):
        start = i * samples_per_realization
        end = (i + 1) * samples_per_realization
        Psi_LPFafter_temp = apply_low_pass_filter(Psi[start:end, :], cutoff_freq, order, sampling_rate, padding_length)     
        pred_mean_LPFafter_temp = apply_low_pass_filter(pred_mean[start:end, :], cutoff_freq, order, sampling_rate, padding_length)
        
        Psi_LPFafter = np.concatenate((Psi_LPFafter, Psi_LPFafter_temp), axis=0)
        pred_mean_LPFafter = np.concatenate((pred_mean_LPFafter, pred_mean_LPFafter_temp), axis=0)

    # Compute R² and correlation
    r2_mean = np.empty(n_outputs)
    r2_mean_LPFafter = np.empty(n_outputs)
    corr_mean = np.empty(n_outputs)
    corr_mean_LPFafter = np.empty(n_outputs)

    for latind in range(n_outputs):
        r2_mean[latind] = r2_score(Psi[:, latind], pred_mean[:, latind])
        r2_mean_LPFafter[latind] = r2_score(Psi_LPFafter[:, latind], pred_mean_LPFafter[:, latind])

        # Pearson correlation
        corr_mean[latind], _ = pearsonr(Psi[:, latind], pred_mean[:, latind])
        corr_mean_LPFafter[latind], _ = pearsonr(Psi_LPFafter[:, latind], pred_mean_LPFafter[:, latind])
    
    # Map results back to (Nlevs, Nlats)
    def map_to_grid(values):
        arr = np.full((Nlevs * Nlats), np.nan)
        arr[Psi_mask] = values
        return arr.reshape((Nlevs, Nlats))

    r2_mean_yz = map_to_grid(r2_mean)
    r2_mean_yz_LPFafter = map_to_grid(r2_mean_LPFafter)
    corr_mean_yz = map_to_grid(corr_mean)
    corr_mean_yz_LPFafter = map_to_grid(corr_mean_LPFafter)

    return (r2_mean_yz, r2_mean_yz_LPFafter,
            corr_mean_yz, corr_mean_yz_LPFafter,
            Psi_LPFafter, pred_mean_LPFafter)





def plot_r2(lat_psi, rho2, r2_mean_yz,  r2_mean_yz_LPFafter, NNpath, Data_str):
    """Plot R² results."""
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    # Plot mean R²
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    pcm = ax.pcolormesh(lat_psi, rho2, r2_mean_yz, cmap='gist_rainbow_r', shading='auto', vmin=0, vmax=1)
    ax.invert_yaxis()
    ax.set_title(r'$R^2$ computed using the mean prediction across all NNs')
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Potential density')
    
    # Format axes
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    # Colorbar
    cbar = fig.colorbar(pcm, ax=ax, orientation='vertical')
    cbar.set_label(r'$R^2$')
    
    plt.savefig(os.path.join(NNpath, f'TestR2_{Data_str}.png'), dpi=300)
    plt.show()
    
    
    # Plot mean R² LPFafter
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    pcm = ax.pcolormesh(lat_psi, rho2, r2_mean_yz_LPFafter, cmap='gist_rainbow_r', shading='auto', vmin=0, vmax=1)
    ax.invert_yaxis()
    ax.set_title(r'$R^2$ computed using the mean prediction across all NNs; 10-year LPF after training')
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Potential density')
    
    # Format axes
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    # Colorbar
    cbar = fig.colorbar(pcm, ax=ax, orientation='vertical')
    cbar.set_label(r'$R^2$')
    
    plt.savefig(os.path.join(NNpath, f'TestR2_{Data_str}_10YLPFafter.png'), dpi=300)
    plt.show()
    

    
def reconstruct_and_plot_hovmoller(Psi, pred_mean, Psi_mask, lat_psi, rho2, year0, output_dir, Data_str):
    """Reconstruct Psi and pred arrays and plot Hovmöller diagrams for selected latitudes."""

    def plot_hovmoller_per_realization(data, rho2, Years, n_realizations, samples_per_realization,
                                        title_prefix, filename_suffix, output_dir, vlim,
                                        latitude=26.5, cmap='RdBu_r'):
        """Plot realization-wise Hovmöller diagram at specified latitude."""
        fig, axes = plt.subplots(
            n_realizations, 1,
            figsize=(12, 2.5 * n_realizations),
            sharex=True,
            constrained_layout=True)
    
        levels = np.linspace(-vlim, vlim, 41)
    
        for i in range(n_realizations):
            ax = axes[i]
            start = i * samples_per_realization
            end = (i + 1) * samples_per_realization
    
            T, R = np.meshgrid(Years, rho2)
            cf = ax.contourf(T, R, data[start:end, :].T,
                             levels=levels, cmap=cmap, extend='both')
            for c in cf.collections:
                c.set_rasterized(True)
            ax.invert_yaxis()
            ax.set_ylabel('Potential density')
            if i == 0:
                ax.set_title(f'{title_prefix} at {latitude:.1f}°N (Realization {i + 36})')
            else:
                ax.set_title(f'Realization {i + 36}')
            ax.set_xlim(Years[0], Years[-1])
    
        axes[-1].set_xlabel('Year')
    
        # Shared colorbar
        cbar = fig.colorbar(cf, ax=axes.ravel().tolist(), orientation='vertical',
                            label='Sv', shrink=1, aspect=50)

        save_name = f'MOC_{latitude:.1f}N_{filename_suffix}_{Data_str}.png'
        plt.savefig(os.path.join(output_dir, save_name), dpi=300)
        plt.show()

    # Reconstruct data arrays
    Nsamps = Psi.shape[0]
    N_flat = Psi.shape[1]
    Nlevs = rho2.shape[0]
    Nlats_psi = lat_psi.shape[0]

    Psi_yz = np.full((Nsamps, Nlevs * Nlats_psi), np.nan)
    Psi_yz[:, Psi_mask] = Psi
    Psi_yz = Psi_yz.reshape((Nsamps, Nlevs, Nlats_psi))
    
    pred_yz = np.full((Nsamps, Nlevs * Nlats_psi), np.nan)
    pred_yz[:, Psi_mask] = pred_mean
    pred_yz = pred_yz.reshape((Nsamps, Nlevs, Nlats_psi))

    # Find latitude index
    def find_nearest_lat_index(lat_array, target_lat):
        """Return the index of the latitude value in `lat_array` closest to `target_lat`."""
        lat_array = np.asarray(lat_array)
        return np.argmin(np.abs(lat_array - target_lat))
    
    Lats_for_plotting = [-55, 0.5, 26.5] 
    # Latitude list
    for latitude in Lats_for_plotting:
        # Get index of closest match
        ind_y = find_nearest_lat_index(lat_psi, latitude)
        
        # Extract along-latitude time series
        Psi_lat = Psi_yz[:, :, ind_y]  # (Nsamps, Nlevs)
        pred_lat = pred_yz[:, :, ind_y]
        
        for ind_lev in range(Nlevs):
            print(r2_score(Psi_lat[:, ind_lev], pred_lat[:, ind_lev]))

        # Time axis
        n_realizations = 5
        samples_per_realization = Nsamps // n_realizations
        Years = np.arange(samples_per_realization) / 12 + year0
        
        # vmax settings
        vmax_truth = np.round(np.abs(Psi_lat).max())
        vmax_error = np.round(vmax_truth / 5)
        
        # Plot truth
        plot_hovmoller_per_realization(
            data=Psi_lat,
            rho2=rho2,
            Years=Years,
            n_realizations=n_realizations,
            samples_per_realization=samples_per_realization,
            title_prefix='Diagnosed MOC',
            filename_suffix='truth',
            output_dir=output_dir,
            vlim=vmax_truth,
            latitude=latitude
        )
        
        # Plot prediction
        plot_hovmoller_per_realization(
            data=pred_lat,
            rho2=rho2,
            Years=Years,
            n_realizations=n_realizations,
            samples_per_realization=samples_per_realization,
            title_prefix='Reconstructed MOC',
            filename_suffix='pred',
            output_dir=output_dir,
            vlim=vmax_truth,
            latitude=latitude
        )
        
        # Plot error
        plot_hovmoller_per_realization(
            data=pred_lat - Psi_lat,
            rho2=rho2,
            Years=Years,
            n_realizations=n_realizations,
            samples_per_realization=samples_per_realization,
            title_prefix='Reconstruction error',
            filename_suffix='error',
            output_dir=output_dir,
            vlim=vmax_error,
            latitude=latitude
        )



def process_CMIP_case(CMIP_name, Data_str, year0):
    """Process a single CMIP case."""
    print(f"\n=== Processing {CMIP_name} ===")

    str_file = '_r36_r40'
    LPF_data_str = '_LPF_ALL' #if LPF_month == 24 else '_ALL'
    if 'ext' in Data_str:
        data_dir = os.path.join(CMIP_DATA_ROOT, CMIP_name, '2100-2300', MOC_str)
    else:
        data_dir = os.path.join(CMIP_DATA_ROOT, CMIP_name, MOC_str)

    require_existing_directory(data_dir, f'Input data for {CMIP_name}')
    require_existing_file(
        os.path.join(data_dir, 'MOC' + str_file + '.npz'),
        f'MOC file for {CMIP_name}',
    )
    require_existing_directory(NNpath, 'Trained neural-network output')

    lat_psi, rho2, Psi, Psi_mask, InputALL, InputNumInd, Nsamps, Nlevs = load_data(
        data_dir, str_file, covariate_names, LPF_data_str, AMOC26_input, AMOC56_input
    )

    pred_mean = load_model_and_predict(NNpath, InputALL, usePCA, Psi_mask, n_folds, n_ensembles)
    

    r2_mean_yz, r2_mean_yz_LPFafter,corr_mean_yz, corr_mean_yz_LPFafter,Psi_LPFafter, pred_mean_LPFafter = compute_r2_and_corr(Psi, pred_mean, Psi_mask, Nlevs, lat_psi.shape[0],5)
    
    variables_dict = {'y_pred': pred_mean, 'y': Psi,'y_pred_10LPF': pred_mean_LPFafter, 'y_10LPF': Psi_LPFafter}
    sio.savemat(os.path.join(NNpath, f'Pred_{Data_str}.mat'), variables_dict)
    
    variables_dict = {'r2_mean_yz': r2_mean_yz, 
                      'r2_mean_yz_LPFafter': r2_mean_yz_LPFafter, 
                      'corr_mean_yz':corr_mean_yz, 
                      'corr_mean_yz_LPFafter':corr_mean_yz_LPFafter,
                      'rho2': rho2, 'lat_psi': lat_psi}
    sio.savemat(os.path.join(NNpath, f'TestR2_{Data_str}.mat'), variables_dict)

    plot_r2(lat_psi, rho2, r2_mean_yz, r2_mean_yz_LPFafter, NNpath, Data_str)

    reconstruct_and_plot_hovmoller(
        Psi=Psi,
        pred_mean=pred_mean,
        Psi_mask=Psi_mask,
        lat_psi=lat_psi,
        rho2=rho2,
        year0=year0,
        output_dir=NNpath,
        Data_str=Data_str
    )

#%% choose neural network
# =========================================================
# 1) Baseline
# =========================================================
baseline_tests = [
    'Test_FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.01Drop0.2_LPF2Year',
]


# =========================================================
# 2) PCA dimension tests
# =========================================================
pca_tests = [
    'FullDepth_PCAinY16_ResNet_Neur512x256x128x64_5foldCV_Reg0.01_LPF2Year',
    # 'FullDepth_PCAinY32_ResNet_Neur512x256x128x64_5foldCV_Reg0.01_LPF2Year',
    # 'FullDepth_PCAinY64_ResNet_Neur512x256x128x64_5foldCV_Reg0.01_LPF2Year',
    # 'FullDepth_PCAinY128_ResNet_Neur512x256x128x64_5foldCV_Reg0.01_LPF2Year',
    # 'FullDepth_ResNet_Neur512x256x128x64_5foldCV_Reg0.01_LPF2Year',
    'FullDepth_PCAinY16_ResNet_Neur512x256x128x64_5foldCV_Reg0.01Drop0.2_LPF2Year',
    'FullDepth_PCAinY32_ResNet_Neur512x256x128x64_5foldCV_Reg0.01Drop0.2_LPF2Year',
    'FullDepth_PCAinY64_ResNet_Neur512x256x128x64_5foldCV_Reg0.01Drop0.2_LPF2Year',
    'FullDepth_PCAinY128_ResNet_Neur512x256x128x64_5foldCV_Reg0.01Drop0.2_LPF2Year',
    'FullDepth_ResNet_Neur512x256x128x64_5foldCV_Reg0.01Drop0.2_LPF2Year',
]


# =========================================================
# 3) Activation-function tests
# =========================================================
activation_tests = [
    # 'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.01_linearActivation_LPF2Year',
    # 'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.01_reluActivation_LPF2Year',
    # 'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.01_geluActivation_LPF2Year',
    # 'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.01_sigmoidActivation_LPF2Year',
    'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.01Drop0.2_linearActivation_LPF2Year',
    'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.01Drop0.2_reluActivation_LPF2Year',
    'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.01Drop0.2_geluActivation_LPF2Year',
    'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.01Drop0.2_sigmoidActivation_LPF2Year',
]

# =========================================================
# 4) Regularization tests
# =========================================================
reg_tests = [
    # 'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0_LPF2Year',
    # 'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.0001_LPF2Year',
    # 'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.001_LPF2Year',
    # 'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.01_LPF2Year',
    # 'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.1_LPF2Year',
]

# =========================================================
# 5) Dropout tests
# =========================================================
dropout_tests = [
    # 'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0Drop0.2_LPF2Year',
    # 'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.0001Drop0.2_LPF2Year',
    # 'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.001Drop0.2_LPF2Year',
    # 'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.01Drop0.2_LPF2Year',
    # 'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.1Drop0.2_LPF2Year',
]

# =========================================================
# 6) Architecture tests
# =========================================================
arch_tests = [
    # 'FullDepth_PCAinY50_ResNet_Neur32_5foldCV_Reg0.01_LPF2Year',
    # 'FullDepth_PCAinY50_ResNet_Neur64_5foldCV_Reg0.01_LPF2Year',
    # 'FullDepth_PCAinY50_ResNet_Neur128x64_5foldCV_Reg0.01_LPF2Year',
    # 'FullDepth_PCAinY50_ResNet_Neur128x64x128_5foldCV_Reg0.01_LPF2Year',
    # 'FullDepth_PCAinY50_ResNet_Neur256_5foldCV_Reg0.01_LPF2Year',
    # 'FullDepth_PCAinY50_ResNet_Neur512_5foldCV_Reg0.01_LPF2Year',
    # 'FullDepth_PCAinY50_ResNet_Neur512x256_5foldCV_Reg0.01_LPF2Year',
    # 'FullDepth_PCAinY50_ResNet_Neur512x256x128_5foldCV_Reg0.01_LPF2Year',
    # 'FullDepth_PCAinY50_ResNet_Neur512x64x512_5foldCV_Reg0.01_LPF2Year',
    'FullDepth_PCAinY50_ResNet_Neur32_5foldCV_Reg0.01Drop0.2_LPF2Year',
    'FullDepth_PCAinY50_ResNet_Neur64_5foldCV_Reg0.01Drop0.2_LPF2Year',
    'FullDepth_PCAinY50_ResNet_Neur128x64_5foldCV_Reg0.01Drop0.2_LPF2Year',
    'FullDepth_PCAinY50_ResNet_Neur128x64x128_5foldCV_Reg0.01Drop0.2_LPF2Year',
    'FullDepth_PCAinY50_ResNet_Neur256_5foldCV_Reg0.01Drop0.2_LPF2Year',
    'FullDepth_PCAinY50_ResNet_Neur512_5foldCV_Reg0.01Drop0.2_LPF2Year',
    'FullDepth_PCAinY50_ResNet_Neur512x256_5foldCV_Reg0.01Drop0.2_LPF2Year',
    'FullDepth_PCAinY50_ResNet_Neur512x256x128_5foldCV_Reg0.01Drop0.2_LPF2Year',
    'FullDepth_PCAinY50_ResNet_Neur512x64x512_5foldCV_Reg0.01Drop0.2_LPF2Year',
]

# =========================================================
# 7) MAE-loss tests
# =========================================================
maeloss_tests = [
    'FullDepth_PCAinY50_ResNet_Neur64_5foldCV_Reg0.01_maeloss_LPF2Year',
    'FullDepth_PCAinY50_ResNet_Neur256_5foldCV_Reg0.01_maeloss_LPF2Year',
    'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.01_maeloss_LPF2Year',
]


experiment_list = (
    baseline_tests
    # pca_tests
    # activation_tests
    # + reg_tests
    # + dropout_tests
    # + arch_tests
    # + maeloss_tests
)

# Update this list when you want to evaluate additional CMIP experiments.
evaluation_cases = [
    ('ACCESS_SSP245', 'SSP245', 2015),
    # ('ACCESS_SSP126', 'SSP126', 2015),
    # ('ACCESS_SSP370', 'SSP370', 2015),
]

for NNstr in experiment_list:
    print(NNstr)
    
    if '_LPF2Year' in NNstr:
        LPFstr = '_LPF2Year' 
    else:
        LPFstr = '' 
    # NNstr = NNstr+LPFstr
    NNpath = os.path.join(NNpath_base, MOC_str, 'results'+LPFstr, NNstr, inputVar)
    require_existing_directory(NNpath, f'Model output for {NNstr}')
    print(NNpath)
    

    n_folds = 5
    n_ensembles = 5
    usePCA = int('PCA' in NNstr)
    
    
    for CMIP_name, Data_str, year0 in evaluation_cases:
        process_CMIP_case(CMIP_name, Data_str, year0)
        
    tf.keras.backend.clear_session()
    gc.collect()
    plt.close('all')
        


