"""
Reconstruct real-world MOC variability from satellite observations on mascons.

"""

import gc
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import r2_score

from _keras_utils import load_model_with_noise_support
from _path_utils import require_existing_directory, require_existing_file
from _runtime_utils import configure_tensorflow_runtime, prepare_covariate_config


configure_tensorflow_runtime()

OBS_DATA_ROOT = r'D:\OneDrive - University of California\PythonWork\mascon'
CMIP_DATA_ROOT = r'E:\Data_CMIP6'
MODEL_BASE_ROOT = r'E:\Analysis2026\ACCESS_hist+SSP585'
REFERENCE_CMIP_NAME = 'ACCESS_SSP245'
USE_ERA5_WINDS = 0
def build_permute_suffix(permute_obp, permute_ssh, permute_uas):
    """Build an output suffix that records which observational inputs were permuted."""

    active_terms = []
    if permute_obp:
        active_terms.append('obp')
    if permute_ssh:
        active_terms.append('ssh')
    if permute_uas:
        active_terms.append('uas')
    return '' if not active_terms else '_permute_' + ''.join(active_terms)


OBS_DATA_ROOT = require_existing_directory(OBS_DATA_ROOT, 'Observation data root')
CMIP_DATA_ROOT = require_existing_directory(CMIP_DATA_ROOT, 'CMIP data root')
MODEL_BASE_ROOT = require_existing_directory(MODEL_BASE_ROOT, 'Model output root')


# ========== User settings ==========
MOC_str = 'ASMOC'

covariate_names = "obp_mascon_V5,ssh_mascon_V5,uas_mascon_V5"


permute_obp = 0
permute_ssh = 0
permute_uas = 0

permute_str = build_permute_suffix(permute_obp, permute_ssh, permute_uas)

# Optional alternative experiments can be enabled by editing the lists below.

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
    #   activation_tests
    # + reg_tests
    # + dropout_tests
    # + arch_tests
    # +maeloss_tests
)

# experiment_list
for NNstr in experiment_list:
    print(NNstr)
    
    # 'FullDepth_ResNet_Neur128x128_5foldCV_Reg0.01Drop0.2'
    # NNstr= 'FullDepth_PCAinY0.95_ResNet_Neur128x128_5foldCV_Reg0.01Drop0.2_geluActivation'
    AMOC26_input = 0
    AMOC56_input = 0
    use_ERA5 = USE_ERA5_WINDS
 
    
    run_config = prepare_covariate_config(covariate_names, AMOC26_input, AMOC56_input)
    covariate_names = run_config.covariate_names
    inputVar = run_config.input_var
    if run_config.noise_size:
        print("NoiseSize =", run_config.noise_size)
    
    
    if '_LPF2Year' in NNstr:
        LPFstr = '_LPF2Year' 
    else:
        LPFstr = '' 
    
    NNpath_base = MODEL_BASE_ROOT


    NNpath = os.path.join(NNpath_base, MOC_str, 'results'+LPFstr, NNstr, inputVar)
    require_existing_directory(NNpath, f'Model output for {NNstr}')
    print(NNpath)
    
    
    n_folds = 5
    n_ensembles = 5
    usePCA = int('PCA' in NNstr)
    
    Nlevs = 18
    Nlats = 140
    # Nsamps = 278
    #%%
    # ========== Helper Functions ==========
    
    def load_data_GRACE():
        grace_file = require_existing_file(
            os.path.join(OBS_DATA_ROOT, MOC_str, 'obp_GRACE.npz'),
            'GRACE mascon data',
        )
        Grace_data = np.load(grace_file)
        if LPFstr == '_LPF2Year':
            obp_GRACE = Grace_data['obp_GRACE_LPF_ALL']
        elif LPFstr == '':
            obp_GRACE = Grace_data['obp_GRACE']
        GRACE_lon = Grace_data['obp_GRACE_lon']
        GRACE_lat = Grace_data['obp_GRACE_lat']
        return obp_GRACE, GRACE_lon,GRACE_lat
    
    
    
    def load_data_DUACS():
        duacs_file = require_existing_file(
            os.path.join(OBS_DATA_ROOT, MOC_str, 'ssh_DUACS.npz'),
            'DUACS mascon data',
        )
        DUACS_data = np.load(duacs_file)
        if LPFstr == '_LPF2Year':
            ssh_DUACS = DUACS_data['ssh_DUACS_LPF_ALL'] # [time lat lon]
        elif LPFstr == '':
            ssh_DUACS = DUACS_data['ssh_DUACS'] # [time lat lon]
        return ssh_DUACS
    
    
    def load_data_uas():
        if(use_ERA5):
            uas_file = require_existing_file(
                os.path.join(OBS_DATA_ROOT, MOC_str, 'uas_ERA5.npz'),
                'ERA5 near-surface zonal wind data',
            )
            ERA5_data = np.load(uas_file)
            if LPFstr == '_LPF2Year':
                uas = ERA5_data['uas_ERA5_LPF_ALL'] # [time lat lon]
            elif LPFstr == '':
                uas = ERA5_data['uas_ERA5'] # [time lat lon]
            uas = uas[3:-9]
        else:
            uas_file = require_existing_file(
                os.path.join(OBS_DATA_ROOT, MOC_str, 'uas_CCMP.npz'),
                'CCMP near-surface zonal wind data',
            )
            CCMP_data = np.load(uas_file)
            if LPFstr == '_LPF2Year':
                uas = CCMP_data['uas_CCMP_LPF_ALL'] # [time lat lon]
            elif LPFstr == '':
                uas = CCMP_data['uas_CCMP'] # [time lat lon]
        return uas
    
    
    
    def load_model_and_predict(NNpath, InputALL, usePCA, n_folds, n_ensembles,Re_norm_str):
        """Load models and predict."""
        Nsamps = InputALL.shape[0]
        n_outputs = Nlevs*Nlats
        pred_ALL = np.empty((n_folds*n_ensembles, Nsamps, n_outputs))
        
        if Re_norm_str == '_Renorm':
            scaler_x = StandardScaler()
            X = scaler_x.fit_transform(InputALL)
            
        ind = 0
        for fold_no in range(1, n_folds+1):
            
            if Re_norm_str == '':
                scaler_x = joblib.load(os.path.join(NNpath, f'scaler_x_fold{fold_no}.pkl'))
                X = scaler_x.transform(InputALL)
                
            scaler_y = joblib.load(os.path.join(NNpath, f'scaler_y_fold{fold_no}.pkl'))
            
            if usePCA:
                pca_y = joblib.load(os.path.join(NNpath, f'pca_y_fold{fold_no}.pkl'))
            for ens_no in range(1, n_ensembles+1):
                model = load_model_with_noise_support(os.path.join(NNpath, f'model_fold{fold_no}_ens{ens_no}.h5'))
                y_pred = model.predict(X, verbose=1)
                if usePCA:
                    y_pred = pca_y.inverse_transform(y_pred)
                pred_ALL[ind] = scaler_y.inverse_transform(y_pred)
                ind = ind+1
        return pred_ALL, Nsamps
    
    #%%
    
    data_dir = require_existing_directory(
        os.path.join(CMIP_DATA_ROOT, REFERENCE_CMIP_NAME, MOC_str),
        f'Reference CMIP data for {REFERENCE_CMIP_NAME}',
    )
    reference_moc_file = require_existing_file(
        os.path.join(data_dir, 'MOC' +'_r36_r40' + '.npz'),
        'Reference MOC file',
    )
    data_MOC = np.load(reference_moc_file)
    rho2 = data_MOC['rho2_full']
    lat_psi=data_MOC['lat_psi']
        
    input_data = np.empty((277,0)) #month,mascon
    
    if("obp_mascon" in covariate_names):
        obp_GRACE, GRACE_lon,GRACE_lat = load_data_GRACE()
        obp_GRACE = obp_GRACE[:-1,:]
        if permute_obp:
            obp_GRACE[:] = np.mean(obp_GRACE, axis=0, keepdims=True) 
        input_data = np.concatenate((input_data,obp_GRACE), axis=1)
        
    
    if("ssh_mascon" in covariate_names):
        ssh_DUACS = load_data_DUACS()
        if permute_ssh:
            ssh_DUACS[:] = np.mean(ssh_DUACS, axis=0, keepdims=True) 
        input_data = np.concatenate((input_data,ssh_DUACS), axis=1)
    
    
    if("uas_mascon" in covariate_names):
        uas = load_data_uas()
        if permute_uas:
            uas[:] = np.mean(uas, axis=0, keepdims=True) 
        input_data = np.concatenate((input_data,uas), axis=1)
        
        
    realworld_output_dir = os.path.join(NNpath, 'RealWorld')
    os.makedirs(realworld_output_dir, exist_ok=True)
    
    
    for Re_norm_str in ['']: #,'_Renorm'
    
        pred_ALL,Nsamps = load_model_and_predict(NNpath, input_data, usePCA, n_folds, n_ensembles,Re_norm_str)
        
        pred_ALLfolds_yz = pred_ALL.reshape((25,Nsamps,Nlevs, Nlats))
        
        pred_mean_over_folds_yz = np.mean(pred_ALLfolds_yz, axis=0)
        pred_yz_std = np.std(pred_ALLfolds_yz, axis=0)
        # pred_mean_over_folds_yz = pred_mean_over_folds.reshape((Nsamps,Nlevs, Nlats))
        
        variables_dict = {'pred_yz': pred_mean_over_folds_yz, 
                          'pred_yz_std': pred_yz_std, 
                          'lat': lat_psi, 
                          'rho2': rho2}

        sio.savemat(os.path.join(realworld_output_dir, 'Pred_RealWorld' + Re_norm_str +permute_str+'.mat'), variables_dict)
        
        
        
        pred_time_mean = np.mean(pred_mean_over_folds_yz,axis=0)
        fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
        pcm = ax.pcolormesh(lat_psi, rho2, pred_time_mean, cmap='RdBu_r', shading='auto', vmin=-20, vmax=20)
        ax.invert_yaxis()
        ax.set_title(r'Reconstructed MOC')
        ax.set_xlabel('Latitude')
        ax.set_ylabel('Potential density')
        # Colorbar
        cbar = fig.colorbar(pcm, ax=ax, orientation='vertical')
        cbar.set_label(r'[Sv]')
        plt.savefig(os.path.join(realworld_output_dir,'Pred_meanMOC_RealWorld'+ Re_norm_str + '.png'), dpi=300)
        plt.show()
    
        pred_time_std = np.std(pred_mean_over_folds_yz,axis=0)
        fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
        pcm = ax.pcolormesh(lat_psi, rho2, pred_time_std, cmap='RdYlBu_r', shading='auto', vmin=0, vmax=5)
        ax.invert_yaxis()
        ax.set_title(r'Reconstructed MOC STD')
        ax.set_xlabel('Latitude')
        ax.set_ylabel('Potential density')
        # Colorbar
        cbar = fig.colorbar(pcm, ax=ax, orientation='vertical')
        cbar.set_label(r'[Sv]')
        plt.savefig(os.path.join(realworld_output_dir,'Pred_STD_RealWorld'+ Re_norm_str + '.png'), dpi=300)
        plt.show()
        
        
            
        var = pred_mean_over_folds_yz
        print("min:", np.nanmin(var))
        print("max:", np.nanmax(var))
        print("mean:", np.nanmean(var))
        print("std:", np.nanstd(var))
        
        tf.keras.backend.clear_session()
        gc.collect()



