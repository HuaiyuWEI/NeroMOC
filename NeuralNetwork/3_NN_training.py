"""
Train the neural networks for MOC reconstruction using CMIP6 data.

"""



import gc
import inspect
import logging
import os
import shutil
import sys
import joblib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import tensorflow as tf
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Activation, Add, Dense, Dropout, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
from matplotlib.colors import Normalize
from scipy.signal import butter, sosfiltfilt
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.layers import PReLU

from _keras_utils import FeaturewiseGaussianNoise
from _path_utils import ensure_directory, require_existing_directory, require_existing_file
from _runtime_utils import configure_tensorflow_runtime, prepare_covariate_config

configure_tensorflow_runtime()

ANALYSIS_ROOT = os.path.join("E:", "Analysis2026")
CMIP_DATA_ROOT = r"E:\Data_CMIP6"
LOGGER_NAME = 'logfile2.log'


def normalize_script_path(path_candidate):
    """Return a usable absolute script path, or ``None`` for interactive placeholders."""

    if not path_candidate:
        return None

    if not isinstance(path_candidate, str):
        path_candidate = str(path_candidate)

    path_candidate = path_candidate.strip()
    if not path_candidate or path_candidate.startswith("<"):
        return None

    normalized_path = os.path.abspath(path_candidate)
    invalid_basenames = {"", "-c", "ipykernel_launcher.py"}
    if os.path.basename(normalized_path) in invalid_basenames:
        return None
    if not os.path.isfile(normalized_path):
        return None
    return normalized_path


def get_running_script_path():
    """Return the active script path when available, or ``None`` in interactive-only runs."""

    candidate_paths = [
        globals().get("__file__"),
        getattr(sys.modules.get("__main__"), "__file__", None),
    ]

    if sys.argv:
        candidate_paths.append(sys.argv[0])

    try:
        candidate_paths.append(inspect.getsourcefile(get_running_script_path))
    except (OSError, TypeError):
        pass

    try:
        candidate_paths.append(inspect.getfile(get_running_script_path))
    except (OSError, TypeError):
        pass

    for path_candidate in candidate_paths:
        normalized_path = normalize_script_path(path_candidate)
        if normalized_path is not None:
            return normalized_path
    return None


def setup_logger(log_dir, logger_name):
    """Create a file-backed logger for long training runs."""

    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    file_handler = logging.FileHandler(os.path.join(log_dir, logger_name))
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setLevel(logging.INFO)
    console_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger, file_handler, console_handler

#%% User settings


# Supported examples:
# 'ACCESS_historical', 'ACCESS_SSP126', 'ACCESS_SSP245', 'ACCESS_SSP370',
# 'ACCESS_SSP585', 'ACCESS_hist+SSP585'
CMIP_name = 'ACCESS_hist+SSP585'

# Low-pass filter configuration
LPF_month = 24 # We used 2-year-LPFed data for training
LPF_data_str = '_LPF_ALL' if LPF_month == 24 else '_ALL'

use_resnet = 1  # dual-branch neural network (DBNN)

# Density-dependent MOC reconstruction
Full_depth_MOC = 1

# Local neural network setting
use_local_inputs = 0 # 1 means we only use input from one specific latitude
use_local_output = 0 # 1 means we only reconstruct MOC at one specific latitude
local_latind = 101  # ACCESS-ESM1-5: 26.5 N


# Neural network hyperparameters
num_folds = 5
NNrepeats = 5
PartialDataTot = 1 # >1 means we only use part of the data for training
PartialDataNum = 1 # this variable specifies which part of the data we will use


loss_function = 'mse'
activation_function = 'leaky_relu'
epoch_max = 2000
batch_size = 600

NN_patience = 30
learning_rate = 0.001  # Adam default
use_scaler_y = 1

scaler_x_minmax = 0
scaler_x_robust = 0

# EOF settings
# for input variables 
PCA_variability_factor = 0  
# ranges from 0 to 1; 0.95 means we will use PCAs that can explain 95% of the variability
PCA_num = 0 # number of PCAs we will use.
# note: PCA_variability_factor and PCA_num cannot be non-zero at the same time

# for MOC
PCA_y_variability_factor = 0 
PCA_y_num = 50
# note: PCA_y_variability_factor and PCA_y_num cannot be non-zero at the same time


OBP_noise = 0 # 1 means we add synthetic ocean bottom pressure noise *during training*
obpNoiseLevel = 0

# Optional diagnostics outputs
save_in_and_out = 0


save_additional_info = 1

# regularization
dropout_rate = 0.2
reg_strength = 0.01

# Predictor settings
NN_neurons = [512,256,128,64]
covariate_names = "obp_mascon_V5,ssh_mascon_V5,uas_mascon_V5"

# whether or not we consider AMOC at 26N or 56N as input variables.
AMOC26_input = 0
AMOC56_input = 0

run_config = prepare_covariate_config(covariate_names, AMOC26_input, AMOC56_input)
covariate_names = run_config.covariate_names
if run_config.noise_size:
    print("NoiseSize =", run_config.noise_size)

#%% Derived paths and logging

analysis_folder = ANALYSIS_ROOT
data_dir_base = require_existing_directory(CMIP_DATA_ROOT, 'CMIP data root')
MOC_str = 'ASMOC'
data_dir = require_existing_directory(
    os.path.join(data_dir_base, CMIP_name, MOC_str),
    f'Input data for {CMIP_name}',
)
result_dir = os.path.join(analysis_folder, CMIP_name, MOC_str)
ensure_directory(result_dir)

logger, file_handler, console_handler = setup_logger(analysis_folder, LOGGER_NAME)
logger.info("Training log initialized")


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        logger.info("Using GPU for training.")
        logger.info(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as exc:
        print(exc)
else:
    logger.info("Using CPU for training.")

logger.info(os.getenv('TF_GPU_ALLOCATOR'))

# %% flags and directories
use_PCA_x = 1 if PCA_variability_factor + PCA_num != 0 else 0
if PCA_variability_factor != 0 and PCA_num != 0:
    raise ValueError(
        "'PCA_variability_factor' and 'PCA_num' cannot be non-zero simultaneously. "
        "Please specify only one to avoid configuration conflicts."
    )
use_PCA_y = 1 if PCA_y_variability_factor + PCA_y_num != 0 else 0
if PCA_y_variability_factor != 0 and PCA_y_num != 0:
    raise ValueError(
        "Both 'PCA_y_variability_factor' and 'PCA_y_num' cannot be non-zero simultaneously. "
        "Please specify only one to avoid configuration conflicts."
    )


# %% Main Execution -- data loading


# load MOC strength
if Full_depth_MOC:
    moc_training_file = require_existing_file(
        os.path.join(data_dir, 'MOC_r1_r35.npz'),
        'MOC training data',
    )
    data_MOC = np.load(moc_training_file)
    rho2 = data_MOC['rho2_full']
    lat_psi=data_MOC['lat_psi']
    Nlats_psi = lat_psi.shape[0]
    Psi = data_MOC['MOC'+LPF_data_str] 
    Psi = np.transpose(Psi, (0, 2, 1)) # Time, Lev, Lat
    Nsamps = Psi.shape[0]
    Nlevs = Psi.shape[1]

    
    
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    pcm = ax.pcolormesh(lat_psi,rho2,np.std(Psi,axis=0), cmap='RdYlBu_r', shading='auto', vmin=0, vmax=5)
    ax.invert_yaxis()
    ax.set_title('Standard deviation of MOC (Sv)')
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Potential density')
    cbar = fig.colorbar(pcm, ax=ax, orientation='vertical')
    cbar.set_label(r'[Sv]')
    plt.savefig(os.path.join(data_dir, 'STD_FullDepth_MOC.png'),dpi=300)
    plt.show()
    
    Psi_mean = np.mean(Psi, axis=0)
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    pcm = ax.pcolormesh(lat_psi, rho2, Psi_mean, cmap='RdBu_r', shading='auto', vmin=-20, vmax=20)
    ax.invert_yaxis()
    ax.set_title('Mean MOC (Sv)')
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Potential density')
    cbar = fig.colorbar(pcm, ax=ax, orientation='vertical')
    cbar.set_label(r'[Sv]')
    plt.savefig(os.path.join(data_dir, 'Mean_FullDepth_MOC.png'),dpi=300)
    plt.show()
    


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
    
    if  use_local_output:
        Psi = Psi[:,:,local_latind]
        Nlats_psi = 1
        local_latind_in = local_latind
        logger.info(f'Using outputs only at latitude {lat_psi[local_latind]:.2f}')
    else:
        Psi = Psi.reshape(Psi.shape[0],Psi.shape[1]*Psi.shape[2])
        
    Psi_mask = ~np.isnan(Psi).any(axis=0)
    Psi = Psi[:,Psi_mask]
        
else:   
    MatFN_MOC = os.path.join(data_dir, 'Psi.mat')
    data_MOC = sio.loadmat(MatFN_MOC)
    Psi = data_MOC['MOCstrength'].T
    Nsamps = Psi.shape[0]
    
    

del data_MOC


    






# load covariates (i.e., input features)
InputNumInd = np.empty((0))
InputALL = np.empty((Nsamps,0))




# load input variables
if covariate_names:
    for name in covariate_names.split(','):
        predictor_file = require_existing_file(
            os.path.join(data_dir, name + '_r1_r35.npz'),
            f'Predictor data for {name}',
        )
        with np.load(predictor_file) as predictor_data:
            temp = predictor_data[name + LPF_data_str]
            mascon_lon = predictor_data['mascon_lon']
            mascon_lat = predictor_data['mascon_lat']

        logger.info('variables on mascons is loaded')
        plt.figure(figsize=(18, 6))
        scatter = plt.scatter(
            mascon_lon,
            mascon_lat,
            c=np.std(temp, axis=0),
            cmap='OrRd',
            s=60,
            edgecolor='k',
        )
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'STD of {name} on mascons')
        plt.colorbar(scatter, label='Data Value')
        plt.grid(True)
        plt.savefig(os.path.join(data_dir, 'STD_' + name  + '.png'),dpi=300)
        plt.show()
        InputALL = np.concatenate((InputALL,temp),axis = 1)
        InputNumInd = np.append(InputNumInd,temp.shape[1])

        del temp
    
    
if AMOC26_input:
    InputALL = np.concatenate((InputALL,Psi_26),axis = 1)
    InputNumInd = np.append(InputNumInd,Psi_26.shape[1])

if AMOC56_input:
    InputALL = np.concatenate((InputALL,Psi_56),axis = 1)
    InputNumInd = np.append(InputNumInd,Psi_56.shape[1])

if PartialDataTot>1:
    Nsamps = int(Nsamps/PartialDataTot)
    Psi = Psi[Nsamps*(PartialDataNum-1):PartialDataNum*Nsamps,:]
    InputALL = InputALL[Nsamps*(PartialDataNum-1):PartialDataNum*Nsamps,:]
    logger.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    logger.info("Use only part of the data for training")
    logger.info(f"Only 1/{PartialDataTot} of the data are used")
    logger.info(f"Time range (month): from {Nsamps*(PartialDataNum-1)} to from {PartialDataNum*Nsamps}")
    logger.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    

    
    
logger.info(f"number of samples in time: {Nsamps}")
logger.info(f"number of input quantities: {len(InputNumInd)}")
logger.info(f"feature number of each input quantity: {InputNumInd}")
logger.info(f"feature number of output quantity: {Psi.shape[1]}")




Years = np.arange(1, Nsamps + 1) / 12
InputNumIndCum = np.cumsum(InputNumInd)
InputNumIndCum = np.insert(InputNumIndCum,0,0)
InputNumIndCum  = InputNumIndCum.astype(int)
InputNumInd  = InputNumInd.astype(int)




#%%

def get_model_family_name():
    """Return the base model-family label used in experiment names."""

    if use_local_inputs:
        return 'LocalNN'
    if use_local_output:
        return 'OneLatPred'
    if use_resnet:
        return 'ResNet'
    return 'MLP'


def get_full_depth_name_prefix():
    """Format the full-depth prefix without obscuring the core model name."""

    if not Full_depth_MOC:
        return ''

    prefix = 'FullDepth_'
    if use_PCA_y:
        prefix += f'{format_pca_tag(use_PCA_x, use_PCA_y)}_'
    return prefix

def format_pca_tag(include_x, include_y):
    """Format the PCA portion of the experiment name."""

    if include_x and include_y:
        return f'PCAinX{PCA_num + PCA_variability_factor}Y{PCA_y_num + PCA_y_variability_factor}'
    if include_y:
        return f'PCAinY{PCA_y_num + PCA_y_variability_factor}'
    if include_x:
        return f'PCAinX{PCA_num + PCA_variability_factor}'
    return ''


def get_scaler_name_prefix():
    """Format the optional scaler-related prefix used in experiment names."""

    prefix = 'NoYScaler_' if not use_scaler_y else ''
    if scaler_x_minmax:
        prefix += 'MinMaxXScaler_'
    if scaler_x_robust:
        prefix += 'RobustXScaler_'
    return prefix


def create_NN_name():
    """Build the experiment name and LPF suffix for the current run settings."""

    model_family_name = get_model_family_name()

    # Preserve the legacy naming pattern so historical result-folder names stay stable.
    inline_pca_tag = ''
    if use_PCA_x or (use_PCA_y and not Full_depth_MOC):
        inline_pca_tag = format_pca_tag(use_PCA_x, use_PCA_y)

    model_core_name = f'{model_family_name}{inline_pca_tag}'

    if PartialDataTot > 1:
        model_core_name += f'_PartialData0{PartialDataNum}of{PartialDataTot}'

    name_stem = f'{get_full_depth_name_prefix()}{model_core_name}'

    if OBP_noise:
        name_stem += f'_obpNoise{obpNoiseLevel}Pa'

    activation_function_str = '_' + activation_function + 'Activation' if activation_function != 'leaky_relu' else ''
    loss_function_str = '_' + str(loss_function) + 'loss' if loss_function != 'mse' else ''
    scaler_str = get_scaler_name_prefix()
    LPF_str = f'_LPF{int(LPF_month / 12)}Year' if LPF_month else ''
    NN_structure_str = 'x'.join(map(str, NN_neurons)) 
    reg_str = f'Reg{reg_strength}' + (f'Drop{dropout_rate}' if dropout_rate != 0 else '')
    batch_size_str =  f'BS{batch_size}_' if batch_size != 600 else ''
    CV_str = f'{num_folds}foldCV_' if num_folds > 1 else ''

    full_name = (
        f"{name_stem}_{scaler_str}Neur{NN_structure_str}_"
        f"{batch_size_str}{CV_str}{reg_str}{loss_function_str}{activation_function_str}{LPF_str}"
    )
    return full_name, LPF_str
NN_name,LPF_str = create_NN_name()

def construct_output_directory(result_dir, NN_name, LPF_str):
    flat_names_list = covariate_names.split(",")
    covariate_names_str = "+".join(flat_names_list)   
    if AMOC26_input:
        covariate_names_str=covariate_names_str+'+AMOC26'
    if AMOC56_input:
        covariate_names_str=covariate_names_str+'+AMOC56'
    if use_local_output:
        Lat_str = f'{lat_psi[local_latind]:.2f}'
        output_dir = os.path.join(result_dir, 'results'+LPF_str, NN_name, Lat_str,covariate_names_str)
    else:
        output_dir = os.path.join(result_dir, 'results'+LPF_str, NN_name, covariate_names_str)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output path: {'created successfully' if os.path.exists(output_dir) else 'already exists'}")
    logger.info(f"The output path is {output_dir}")
    return output_dir, covariate_names_str,flat_names_list
output_dir, covariate_names_str,flat_names_list = construct_output_directory(result_dir, NN_name,LPF_str)



#%% Define the neural network

# Make another logfile to store loss during training
log_path =os.path.join(output_dir, 'training_logs.txt')
os.makedirs(os.path.dirname(log_path), exist_ok=True)
class PrintTrainingOnTextEvery10EpochsCallback(Callback):
    def __init__(self, log_path):
        super().__init__()
        self.log_path = log_path

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:  # Log every 10 epochs
            with open(self.log_path, "a") as log_file:
                log_file.write(
                    f"Epoch: {epoch:>3} | "
                    f"Loss: {logs.get('loss', 0):.2e} | "
                    f"Validation loss: {logs.get('val_loss', 0):.2e} |\n "
                )
                print(
                    f"Epoch {epoch:>3} - "
                    f"Loss: {logs.get('loss', 0):.2e}, "
                    f"Validation loss: {logs.get('val_loss', 0):.2e}, "
                )

my_callbacks = [
    PrintTrainingOnTextEvery10EpochsCallback(log_path=log_path),
]   

def get_activation(activation_function):
    if activation_function == 'leaky_relu':
        return LeakyReLU(alpha=0.2)
    elif activation_function == 'relu':
        return 'relu'
    elif activation_function == 'sigmoid':
        return 'sigmoid'
    elif activation_function == 'tanh':
        return 'tanh'
    elif activation_function == 'elu':
        return 'elu'
    elif activation_function == 'linear':
        return 'linear'
    elif activation_function == 'prelu':
        print('PreLu activation layer is added!')
    elif activation_function == 'gelu':
        return 'gelu'
    else:
        raise ValueError(f"Unsupported activation function: {activation_function}")


def gpu_memory():
    memory_info = tf.config.experimental.get_memory_info('GPU:0')
    memory_info = memory_info['current'] / (1024 ** 2)# Convert from bytes to MiB
    logging.info(f'TensorFlow memory usage: {memory_info:.2f} MiB')
    print(f'TensorFlow memory usage: {memory_info:.2f} MiB')
    return  memory_info 

#######################################################################
#                       define the neural network                     #
#######################################################################


def train_model(X_all,y_all,X_train, y_train, X_test, y_test,scaler_y,scaler_x):
    
    activation = get_activation(activation_function)
    # Create a new model with random initial weights and biases
    inputs_raw = tf.keras.Input(shape=(X_train.shape[1],))
    
    if OBP_noise:
        stddev_norm = obpNoiseLevel / scaler_x.scale_   # shape=(n_features,)
        inputs= FeaturewiseGaussianNoise(stddev=stddev_norm)(inputs_raw)
        logger.info("Gaussian noise has been added to OBP")
        logger.info(f'stddev_norm[:10] = {stddev_norm[:10]}')
    else:
        inputs=inputs_raw
        

    output1 = Dense(units=NN_neurons[0],
                       kernel_regularizer=regularizers.l2(reg_strength))(inputs)
    
    if activation_function == 'prelu':
        output = PReLU()(output1)
    else:
        output = Activation(activation)(output1)
    
    if dropout_rate!=0:
        output = Dropout(dropout_rate)(output)

    # Add more dense layers with non-linear activation 
    if len(NN_neurons) > 1:
        for i in range(1, len(NN_neurons)):

            if activation_function == 'prelu':
                output = Dense(units=NN_neurons[i],
                               kernel_regularizer=regularizers.l2(reg_strength))(output)
                output = PReLU()(output)
            else:
                output = Dense(units=NN_neurons[i], activation=activation,
                               kernel_regularizer=regularizers.l2(reg_strength))(output)
            
            if dropout_rate!=0:
                output = Dropout(dropout_rate)(output)
                
    
    if use_resnet:
        if dropout_rate!=0:
            output_skip = Dropout(dropout_rate)(output1)                  
        else:
            output_skip = output1
            
        output_skip = Dense(units=y_train.shape[1], activation='linear',
                           kernel_regularizer=regularizers.l2(reg_strength))(output_skip)
                       
        output = Dense(units=y_train.shape[1], activation='linear')(output)
        
        output = Add(name='add_layer')([output,output_skip])
    else:
        # Add the final linear output layer
        output = Dense(units=y_train.shape[1], activation='linear')(output)
    
    
    # Create the model
    model = Model(inputs=inputs_raw, outputs=output)
    
    if fold_no + ens_no == 2:
        # show the model summary
        model.summary()
        with open(os.path.join(output_dir, 'model_' + covariate_names_str  + '.txt'), 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        dot_img_file = os.path.join(output_dir, 'model_' + covariate_names_str  + '.png')
        plot_model(model, to_file=dot_img_file,
                   show_shapes=True,
                   show_dtype=False,
                   show_layer_names=True,
                   rankdir='LR',
                   expand_nested=False,
                   dpi=300,
                   show_layer_activations=True)
        
    
    # Compile the model with mean squared error loss and Adam optimizer
    model.compile(loss=loss_function, optimizer=Adam(learning_rate=learning_rate))
    
    # Define early stopping callback
    early_stopping = EarlyStopping(patience=NN_patience, monitor='val_loss', mode='min', restore_best_weights=1, verbose=1)
    gpu_memory()
    # Train the model with early stopping callback
    history = model.fit(X_train, y_train, epochs=epoch_max, batch_size=batch_size,
                        validation_data=(X_test, y_test), callbacks=[early_stopping,my_callbacks],verbose=0)
    gpu_memory()
    # Evaluate the performance on the testing set (less useful)
    skill = model.evaluate(X_test, y_test, verbose=0) 
    
        
    # Make the prediction with the actual scale in the testing set
    if use_PCA_y:
        pred = model.predict(X_test)
    elif num_folds<=1:
        pred = scaler_y.inverse_transform(model.predict(X_all))
    else:
        pred = scaler_y.inverse_transform(model.predict(X_test))
    
    
    # Calculate the R2 value in the testing set at each latitude 
    R2_AllLats = []
    if use_PCA_y:
        truth = y_test
    elif num_folds<=1:
        truth = scaler_y.inverse_transform(y_all)
    else:
        truth = scaler_y.inverse_transform(y_test)
        
    for latind in range(y_test.shape[1]):
        y_lat = truth[:, latind];
        y_pred_lat = pred[:, latind];
        R2_AllLats.append(r2_score(y_lat, y_pred_lat))
        
    model.save(os.path.join(output_dir,'model_fold'+str(fold_no)+'_ens'+str(ens_no)+'.h5'))

    return skill, history, pred, R2_AllLats


#%% 
#######################################################################
#                             Train loop                              #
#######################################################################
X = InputALL
y = Psi

print(X.shape)
print(y.shape)

    
sio.savemat(os.path.join(output_dir,'inputs_info.mat'),
            {'mascon_lon': mascon_lon, 'mascon_lat': mascon_lat, 
             'Nsamps':Nsamps,
             'lat_psi':lat_psi,
             'InputNumIndCum':InputNumIndCum,
             'covariate_names':covariate_names})

        




trained_models = []
y_pred_reconstructed_allfolds = []
scaler_y_allfolds = []


# Train the neural network multiple times using k-fold cross validation

# Define the K-fold Cross Validator
if num_folds > 1:
    kfold = KFold(n_splits=num_folds, shuffle=False)
    splits = list(kfold.split(X, y))  # convert to list to index easily


for fold_no in range(1,num_folds+1):
    print(fold_no)
    if num_folds > 1:
        trainind, testind = splits[fold_no-1]
    else:
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        split_point = int(np.round(X.shape[0] * 2 / 3))
        trainind = indices[:split_point]
        testind = indices[split_point:]

    X_train, y_train = X[trainind, :], y[trainind, :]
    X_test, y_test = X[testind, :], y[testind, :]

    # Normalize the input and out data based on the training set
    if scaler_x_minmax:
        scaler_x = MinMaxScaler()
        X_train = scaler_x.fit_transform(X_train)
        X_test = scaler_x.fit_transform(X_test) ## data leakage alert! test purpose only 
    elif scaler_x_robust:
        scaler_x = RobustScaler()
        X_train = scaler_x.fit_transform(X_train)
        X_test = scaler_x.transform(X_test)
    else:
        scaler_x = StandardScaler()
        X_train = scaler_x.fit_transform(X_train)
        X_test = scaler_x.transform(X_test)

    if use_scaler_y:
        scaler_y = StandardScaler()
    else:
        scaler_y = FunctionTransformer()
        
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)
    
    X_all = scaler_x.transform(X)
    y_all = scaler_y.transform(y)
    
    #save the scalers
    joblib.dump(scaler_x, os.path.join(output_dir,f'scaler_x_fold{fold_no}.pkl'))
    joblib.dump(scaler_y, os.path.join(output_dir,f'scaler_y_fold{fold_no}.pkl'))
    
#######################################################################
#                   Use PCA to reduce dimensionality                  #
#######################################################################
    if use_PCA_x: 
        temp3 = np.empty((len(trainind),0))
        temp4 = np.empty((len(testind),0))
        logger.info('------------------------------------------------------------------------')
        logger.info('Computing principal components')
        n = InputNumIndCum
        for i in np.arange(1,len(n)):
            temp = X_train[:, n[i-1]:n[i]]
            if PCA_num!=0:
                pca = PCA(n_components=PCA_num)
                temp = pca.fit_transform(temp)
                pca_explained_variance = np.cumsum(pca.explained_variance_ratio_)*100
                logger.info(f'Adopting the first {PCA_num} princple components of input {i}')
                formatted_variance = ", ".join(f"{var:.2f}" for var in pca_explained_variance)
                logger.info('Cumulative explained variance:')
                logger.info(f'{formatted_variance}')
                logger.info(f"Dimensionality reduction: {InputNumInd[i-1]} --> {temp.shape[1]}")
            elif PCA_variability_factor!=0:
                pca = PCA(n_components=PCA_variability_factor)
                temp = pca.fit_transform(temp)
                
                logger.info(f'Adopting components that explain {PCA_variability_factor*100}$\%$ of the varibility of input {i}')
                logger.info(f"Dimensionality reduction: {InputNumInd[i-1]} --> {temp.shape[1]}")
                
            temp2 = pca.transform(X_test[:, n[i-1]:n[i]])
            temp3 = np.concatenate((temp3,temp),axis = 1)
            temp4 = np.concatenate((temp4,temp2),axis = 1)
    
    if use_PCA_y: 
        logger.info('------------------------------------------------------------------------')
        logger.info('Computing principal components of MOC')
        if PCA_y_num!=0:
            pca_y = PCA(n_components=PCA_y_num)
            y_train = pca_y.fit_transform(y_train)
            pca_explained_variance = np.cumsum(pca_y.explained_variance_ratio_)*100
            logger.info(f'Adopting the first {PCA_y_num} princple components')
            formatted_variance = ", ".join(f"{var:.2f}" for var in pca_explained_variance)
            logger.info('Cumulative explained variance:')
            logger.info(f'{formatted_variance}')
            logger.info(f"Dimensionality reduction: {Psi.shape[1]} --> {y_train.shape[1]}")
            
        elif PCA_y_variability_factor!=0:
            pca_y = PCA(n_components=PCA_y_variability_factor)
            y_train = pca_y.fit_transform(y_train)
            logger.info(f'Adopting components that explain {PCA_y_variability_factor*100}$\%$ of the varibility')
            logger.info(f"Dimensionality reduction: {Psi.shape[1]} --> {y_train.shape[1]}")

        #save the PCAs
        joblib.dump(pca_y, os.path.join(output_dir,f'pca_y_fold{fold_no}.pkl'))
        
        y_test = pca_y.transform(y_test)
        
        scaler_y_allfolds.append(scaler_y)
        if Full_depth_MOC:
            # plot the spatial pattern of the first few principal components
            eof_patterns = np.full((y_train.shape[1], Nlats_psi * Nlevs), np.nan)  # Initialize with NaN
            eof_patterns[:, Psi_mask] = pca_y.components_  # Only fill valid locations
            eof_patterns = eof_patterns.reshape((y_train.shape[1], Nlevs,Nlats_psi))

            PCA_num_plot = min(y_train.shape[1],6)
            max_val = np.nanmax(np.abs(eof_patterns[0:PCA_num_plot]))
            # Create a symmetric colormap around zero
            norm = Normalize(vmin=-max_val, vmax=max_val)
            fig, axes = plt.subplots(nrows=round(PCA_num_plot), ncols=1, figsize=(18, 18))
            ymax = 1037.31
            for i, ax in enumerate(axes.flat):
                if i < PCA_num_plot:  # Ensure we only plot for the number of PCA components
                    im = ax.pcolormesh(lat_psi,rho2,eof_patterns[i], cmap='RdBu_r', norm=norm, shading='auto')
                    explained_variance_pct = pca_y.explained_variance_ratio_[i] * 100  # Convert to percentage
                    title = f"EOF {i+1} ({explained_variance_pct:.2f}%)"
                    ax.set_title(title)
                    ax.set_xlabel('Longitude')
                    ax.set_ylabel('Latitude')
                    ax.set_ylim(1035.25, ymax)
                    ax.invert_yaxis()
                    fig.colorbar(im, ax=ax, orientation='vertical')
                else:
                    ax.axis('off')  # Turn off unused axes
            plt.subplots_adjust(hspace=0.6)  # Adjust vertical spacing between rows
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'PCAs_MOC_fold'+str(fold_no)+'.png'),dpi=300)
            plt.show()
        else:
            # plot the spatial pattern of the first few principal components
            eof_patterns = pca_y.components_  # Initialize with NaN
            PCA_num_plot = min(y_train.shape[1],6)
            max_val = np.nanmax(np.abs(eof_patterns[0:PCA_num_plot]))
            # Create a symmetric colormap around zero                   
            fig, axes = plt.subplots(nrows=round(PCA_num_plot), ncols=1, figsize=(18, 18))
            for i, ax in enumerate(axes.flat):
                if i < PCA_num_plot:  # Ensure we only plot for the number of PCA components
                    im = ax.plot(lat_psi,eof_patterns[i])
                    explained_variance_pct = pca_y.explained_variance_ratio_[i] * 100  # Convert to percentage
                    title = f"EOF {i+1} ({explained_variance_pct:.2f}%)"
                    ax.set_title(title)
                    ax.set_xlabel('Latitude')
                    ax.set_ylabel('EOF magnitude')
                else:
                    ax.axis('off')  # Turn off unused axes
            plt.subplots_adjust(hspace=0.6)  # Adjust vertical spacing between rows
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'PCAs_MOC_fold'+str(fold_no)+'.png'),dpi=300)
            plt.show()             

#######################################################################
#                                PCA ENDs                             #
#######################################################################

    # Generate a print
    logger.info('------------------------------------------------------------------------')
    
    if save_in_and_out:
        variables_dict = {'X_train': X_train, 'y_train': y_train, 'X_test':X_test, 'y_test':y_test}
        # Save variables to a .mat file
        import mat73
        mat73.savemat(os.path.join(output_dir,'model_in_and_out'+str(fold_no) + '.mat'), variables_dict)
        # sio.savemat(os.path.join(output_dir,'model_in_and_out'+str(fold_no) + '.mat'), variables_dict)
        
    if save_additional_info:
        corr_matrix = np.corrcoef(X_train.T)
        
        plt.figure(figsize=(12, 6))
        im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(im, label='Correlation')
        plt.title('Feature Cross-Correlation Matrix')
        plt.xlabel('Feature Index')
        plt.ylabel('Feature Index')
        plt.show()
        
        sio.savemat(os.path.join(output_dir,'inputs_corr.mat'),
                    {'corr_matrix': corr_matrix, 'Nsamps':Nsamps,
                     'InputNumIndCum':InputNumIndCum,
                     'covariate_names':covariate_names})



    # We use ensemble training for each fold of the cross-validation
    for ens_no  in np.arange(1,NNrepeats+1):
        logger.info(f'Training for fold {fold_no} ensemble {ens_no}...')
        skill, history, pred, R2_AllLats = train_model(X_all,y_all,X_train, y_train, X_test, y_test, scaler_y, scaler_x)
        trained_models.append((skill, skill, history.history, pred, R2_AllLats))
        if use_PCA_y:
            y_pred_reconstructed = pca_y.inverse_transform(pred)
            y_pred_reconstructed_allfolds.append(y_pred_reconstructed)



    gpu_memory()
    tf.keras.backend.clear_session()
    gpu_memory()
    gc.collect()
    gpu_memory()
        
        
logger.info('------------------------------------------------------------------------')
logger.info(f'Training with {num_folds}-fold cross-validation finished!') 
logger.info('------------------------------------------------------------------------')




#%% Check if the model skill varies too much among different emsembles and folds
skills = [trained_model[1] for trained_model in trained_models[0:num_folds * NNrepeats]]
skills = np.array(skills)  # convert to NumPy array for numerical ops
median_skill = np.median(skills)
median_index = np.argmin(np.abs(skills - median_skill))
minimum_skill = np.min(skills)
min_index = np.argmin(skills)
std_dev_skill = np.std(skills)

formatted_skills = [f"{skill:.4f}" for skill in skills]
logger.info(f"Losses in testing set: {formatted_skills}")
fold_median = median_index // NNrepeats + 1  # Compute fold number
ensemble_median = median_index % NNrepeats + 1  # Compute ensemble number
logger.info(f"Median Loss is: {median_skill:.4f}, which occurs at Fold {fold_median} Ensemble {ensemble_median}")
fold_min = min_index // NNrepeats + 1  # Compute fold number
ensemble_min = min_index % NNrepeats + 1  # Compute ensemble number
logger.info(f"Minimum Loss is: {minimum_skill:.4f}, which occurs at Fold {fold_min} Ensemble {ensemble_min}")
logger.info(f"Standard Deviation of the Loss is: {std_dev_skill:.4f}")
ratio = std_dev_skill/median_skill * 100
logger.info(f"Standard Deviation/Median: {ratio:.2f}%")

# Check if the standard deviation to median skill ratio is higher than 30%
if ratio > 30:
    warning_message = ("Warning: The standard deviation of model skill across "
                       "folds and ensembles is large!\nStandard Deviation/Median Skill: {ratio:.2f}%").format(ratio=ratio)
    with open(os.path.join(output_dir,'Warning_'+covariate_names_str+'.txt'), "w") as file:
        file.write(warning_message)
    with open(os.path.join(output_dir,'..','Warning_'+covariate_names_str+'.txt'), "w") as file:
        file.write(warning_message)


# Plot the loss function during training for all folds
# First, find the global minimum and maximum loss values across all folds
min_loss = min(min(trained_models[i][2]['loss']+trained_models[i][2]['val_loss']) for i in range(num_folds))
max_loss = max(max(trained_models[i][2]['loss']+trained_models[i][2]['val_loss']) for i in range(num_folds))

plt.figure(figsize=(9, 12))

# Plot the first fold outside the loop to avoid repeating the legend setting
plt.subplot(num_folds, 1, 1)
for nn in range(NNrepeats):
    plt.plot(trained_models[nn][2]['loss'],'-k')
    plt.plot(trained_models[nn][2]['val_loss'],'-r')
plt.ylabel('Loss')
plt.yscale('log')  # Set logarithmic scale
plt.ylim(min_loss, max_loss)  # Set the same y-limits for all subplots
plt.legend(['Train', 'Validation'], loc='upper right')

# Now plot the remaining folds
for n in range(1, num_folds):
    plt.subplot(num_folds, 1, n+1)
    for nn in range(NNrepeats):
        plt.plot(trained_models[n*NNrepeats+nn][2]['loss'],'-k')
        plt.plot(trained_models[n*NNrepeats+nn][2]['val_loss'],'-r')
    plt.ylabel('Loss')
    plt.yscale('log')  # Set logarithmic scale
    plt.ylim(min_loss, max_loss)  # Set the same y-limits for all subplots

plt.xlabel('Epochs')
plt.subplots_adjust(hspace=0.5)  # Adjust space between plots if needed
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'TrainingLoss_' + covariate_names_str  + '.png'),dpi=300)
plt.show()


if Full_depth_MOC:
    sio.savemat(os.path.join(output_dir, 'Psi_mask.mat'), {'Psi_mask': Psi_mask})
    


#%% copy this script to the directory that stores the trained NN
def copy_script(target_directory):
    current_script = get_running_script_path()

    if current_script is None:
        logger.warning("Could not determine the current script path. Skipping script copy.")
        return None
    
    # Ensure the target directory exists, create if it does not
    os.makedirs(target_directory, exist_ok=True)
    
    # Define the target path for the script
    target_path = os.path.join(target_directory, os.path.basename(current_script))
    
    # Copy the script
    shutil.copy(current_script, target_path)
    logger.info(f"Script copied to {target_path}")
    return target_path

# Example usage
copy_script(output_dir)
    
# Copy the logfile
shutil.copy(os.path.join(analysis_folder, LOGGER_NAME), output_dir)
    

if file_handler:
    logger.removeHandler(file_handler)
    logger.removeHandler(console_handler)
    file_handler.close()  
if os.path.exists(os.path.join(analysis_folder, LOGGER_NAME)):
    os.remove(os.path.join(analysis_folder, LOGGER_NAME))


