import os
from dataclasses import dataclass
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import tensorflow as tf
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import r2_score
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

from _keras_utils import load_model_with_noise_support
from _path_utils import require_existing_directory, require_existing_file
from _runtime_utils import configure_tensorflow_runtime, prepare_covariate_config

configure_tensorflow_runtime()


# ========== Basic Configuration ==========
DEFAULT_COVARIATE_NAMES = "obp_mascon_V5,ssh_mascon_V5,uas_mascon_V5" #obp_mascon_V5,ssh_mascon_V5,
AMOC26_INPUT = 0
AMOC56_INPUT = 0

MOC_STR = "ASMOC"
LPF_STR = "_LPF2Year"
LPF_MONTH = 24
LPF_DATA_SUFFIX = "_LPF_ALL" if LPF_MONTH == 24 else "_ALL"
STR_FILE = "_r36_r40"

P_NOISE = 40
SSH_NOISE = 1e-2
NOISE_INTER_NUM = 500
N_FOLDS = 5
N_ENSEMBLES = 5
N_REALIZATIONS = 5

LPF_MONTH_AFTER = 120  # 10-year low-pass filter on monthly data
SAMPLING_RATE = 1
CUTOFF_FREQ = 1 / LPF_MONTH_AFTER
FILTER_ORDER = 5
PADDING_LENGTH = 2 * LPF_MONTH_AFTER

CMIP_DATA_ROOT = r"E:\Data_CMIP6"
MODEL_BASE_PATHS = [
    r"E:\Analysis2026\ACCESS_hist+SSP585",
]
NNSTR_LIST = [
    "Test_FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.01Drop0.2",
]
CMIP_CASES = [
    ("ACCESS_SSP245", "SSP245"),
]

# Add or remove covariate combinations here for batch Monte Carlo runs.
COVARIATE_NAMES_LIST = [
    DEFAULT_COVARIATE_NAMES,
    # "obp_mascon_V5",
    # "ssh_mascon_V5",
    # "obp_mascon_V5,ssh_mascon_V5",
    # "obp_mascon_V5,uas_mascon_V5",
    # "ssh_mascon_V5,uas_mascon_V5",
]
CMIP_DATA_ROOT = require_existing_directory(CMIP_DATA_ROOT, "CMIP data root")


@dataclass(frozen=True)
class RunConfig:
    covariate_names: str
    input_var: str
    noise_size: int


@dataclass
class LoadedData:
    lat_psi: np.ndarray
    rho2: np.ndarray
    psi: np.ndarray
    psi_mask: np.ndarray
    input_all: np.ndarray
    input_num_ind: np.ndarray
    n_samples: int
    n_levels: int
    n_lats: int


@dataclass
class MonteCarloOutput:
    pred_mean: np.ndarray
    pred_mean_yz: np.ndarray
    pred_variance_yz: np.ndarray
    pred_mean_lpf: np.ndarray
    pred_mean_yz_lpf: np.ndarray
    pred_variance_yz_lpf: np.ndarray
    r2_each_mc: np.ndarray


def prepare_run_config(covariate_names, amoc26_input=0, amoc56_input=0):
    shared_config = prepare_covariate_config(
        covariate_names,
        amoc26_input,
        amoc56_input,
    )

    return RunConfig(
        covariate_names=shared_config.covariate_names,
        input_var=shared_config.input_var,
        noise_size=shared_config.noise_size,
    )


# ========== Custom Keras Layer ==========
class FeaturewiseGaussianNoise(Layer):
    """Add Gaussian noise to each feature with separate standard deviations."""

    def __init__(self, stddev, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

        if isinstance(stddev, np.ndarray):
            self.stddev = stddev.tolist()
        elif isinstance(stddev, (list, tuple)):
            self.stddev = list(stddev)
        else:
            self.stddev = [float(stddev)]
        self.stddev_tensor = None

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError("FeaturewiseGaussianNoise expects 2D input (batch_size, features)")

        n_features = input_shape[1]
        if len(self.stddev) == 1:
            self.stddev = self.stddev * n_features
        elif len(self.stddev) != n_features:
            raise ValueError(
                f"Length of stddev ({len(self.stddev)}) must match number of features ({n_features})"
            )

        self.stddev_tensor = tf.constant(self.stddev, dtype=K.floatx())

    def call(self, inputs, training=None):
        def noised():
            noise = tf.random.normal(
                shape=tf.shape(inputs),
                mean=0.0,
                stddev=self.stddev_tensor,
                dtype=inputs.dtype,
            )
            return inputs + noise

        return K.in_train_phase(noised, inputs, training=training)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"stddev": self.stddev})
        return config


# ========== Filtering Helpers ==========
def apply_low_pass_filter(data, cutoff_freq, order=5, sampling_rate=1, padding_length=None):
    """Apply a Butterworth low-pass filter along the time axis."""
    sos = butter(order, cutoff_freq, btype="low", output="sos", analog=False, fs=sampling_rate)
    padding_length = 0 if padding_length is None else padding_length

    if data.ndim == 2:
        _, n_features = data.shape
        filtered_data = np.empty_like(data)
        for feature_idx in range(n_features):
            ts = data[:, feature_idx]
            if np.any(np.isnan(ts)):
                filtered_data[:, feature_idx] = ts
                continue

            padded = np.pad(ts, (padding_length, padding_length), mode="reflect") if padding_length else ts
            filtered = sosfiltfilt(sos, padded)
            filtered_data[:, feature_idx] = filtered[padding_length:-padding_length] if padding_length else filtered
        return filtered_data

    if data.ndim == 3:
        _, n_levels, n_lats = data.shape
        filtered_data = np.empty_like(data)
        for lat_idx in range(n_lats):
            for level_idx in range(n_levels):
                ts = data[:, level_idx, lat_idx]
                if np.any(np.isnan(ts)):
                    filtered_data[:, level_idx, lat_idx] = ts
                    continue

                padded = np.pad(ts, (padding_length, padding_length), mode="reflect") if padding_length else ts
                filtered = sosfiltfilt(sos, padded)
                filtered_data[:, level_idx, lat_idx] = (
                    filtered[padding_length:-padding_length] if padding_length else filtered
                )
        return filtered_data

    raise ValueError("Input data must be 2D or 3D with time as the first dimension.")


def apply_filter_by_realization(data, n_realizations=N_REALIZATIONS):
    """Apply the low-pass filter separately to each realization block."""
    filtered_chunks = []
    n_samples = data.shape[0]
    samples_per_realization = n_samples // n_realizations

    for realization_idx in range(n_realizations):
        start = realization_idx * samples_per_realization
        end = (realization_idx + 1) * samples_per_realization if realization_idx < n_realizations - 1 else n_samples
        if start >= end:
            continue
        filtered_chunks.append(
            apply_low_pass_filter(
                data[start:end],
                cutoff_freq=CUTOFF_FREQ,
                order=FILTER_ORDER,
                sampling_rate=SAMPLING_RATE,
                padding_length=PADDING_LENGTH,
            )
        )

    return np.concatenate(filtered_chunks, axis=0)


def reshape_masked_field(data_2d, mask, n_levels, n_lats):
    full_field = np.full((data_2d.shape[0], n_levels * n_lats), np.nan)
    full_field[:, mask] = data_2d
    return full_field.reshape((data_2d.shape[0], n_levels, n_lats))


def reshape_masked_vector(data_1d, mask, n_levels, n_lats):
    full_field = np.full(n_levels * n_lats, np.nan)
    full_field[mask] = data_1d
    return full_field.reshape((n_levels, n_lats))


# ========== Data Helpers ==========
def find_nearest_lat_index(lat_array, target_lat):
    lat_array = np.asarray(lat_array)
    return np.argmin(np.abs(lat_array - target_lat))


def resolve_data_dir(cmip_name, data_str):
    if "ext" in data_str:
        path = os.path.join(CMIP_DATA_ROOT, cmip_name, "2100-2300", MOC_STR)
    else:
        path = os.path.join(CMIP_DATA_ROOT, cmip_name, MOC_STR)
    return require_existing_directory(path, f"Input data for {cmip_name}")


def load_input_variable(data_dir, name, str_file, lpf_data_suffix, n_samples, noise_size):
    if "whiteNoise" in name:
        if noise_size <= 0:
            raise ValueError(f"Could not determine white noise size from covariate name: {name}")
        return np.zeros((n_samples, noise_size))

    filename = require_existing_file(
        os.path.join(data_dir, f"{name}{str_file}.npz"),
        f"Predictor file for {name}",
    )
    with np.load(filename) as data:
        temp = data[f"{name}{lpf_data_suffix}"]

    if temp.ndim > 2:
        temp = temp.reshape(temp.shape[0], temp.shape[1] * temp.shape[2])
        temp = temp[:, ~np.isnan(temp).any(axis=0)]

    if temp.ndim == 1:
        temp = temp[:, np.newaxis]

    return temp


def load_data(data_dir, str_file, run_config, lpf_data_suffix, amoc26_input, amoc56_input):
    """Load input variables and target MOC strength."""
    moc_file = require_existing_file(
        os.path.join(data_dir, f"MOC{str_file}.npz"),
        "MOC target file",
    )
    with np.load(moc_file) as data_moc:
        rho2 = data_moc["rho2_full"]
        lat_psi = data_moc["lat_psi"]
        psi = np.transpose(data_moc[f"MOC{lpf_data_suffix}"], (0, 2, 1))

    n_samples, n_levels, n_lats = psi.shape

    if amoc26_input:
        psi_26 = psi[:, :, find_nearest_lat_index(lat_psi, 26.5)]
    if amoc56_input:
        psi_56 = psi[:, :, find_nearest_lat_index(lat_psi, 56.5)]

    psi = psi.reshape(n_samples, n_levels * n_lats)
    psi_mask = ~np.isnan(psi).any(axis=0)
    psi = psi[:, psi_mask]

    input_blocks = []
    input_sizes = []

    if run_config.covariate_names:
        for name in run_config.covariate_names.split(","):
            temp = load_input_variable(
                data_dir,
                name,
                str_file,
                lpf_data_suffix,
                n_samples,
                run_config.noise_size,
            )
            input_blocks.append(temp)
            input_sizes.append(temp.shape[1])

    if amoc26_input:
        input_blocks.append(psi_26)
        input_sizes.append(psi_26.shape[1])

    if amoc56_input:
        input_blocks.append(psi_56)
        input_sizes.append(psi_56.shape[1])

    input_all = np.concatenate(input_blocks, axis=1) if input_blocks else np.empty((n_samples, 0))
    input_num_ind = np.asarray(input_sizes, dtype=int)

    return LoadedData(
        lat_psi=lat_psi,
        rho2=rho2,
        psi=psi,
        psi_mask=psi_mask,
        input_all=input_all,
        input_num_ind=input_num_ind,
        n_samples=n_samples,
        n_levels=n_levels,
        n_lats=n_lats,
    )


# ========== Monte Carlo Helpers ==========
def build_feature_noise_vector(covariate_names, input_num_ind, n_features, p_noise, ssh_noise):
    rmse_vec = np.zeros(n_features)
    feature_start = 0

    covariate_list = [name for name in covariate_names.split(",") if name]
    for name, block_size in zip(covariate_list, input_num_ind):
        feature_stop = feature_start + int(block_size)
        if name.startswith("obp_mascon"):
            rmse_vec[feature_start:feature_stop] = p_noise
        elif name.startswith("ssh_mascon"):
            rmse_vec[feature_start:feature_stop] = ssh_noise
        feature_start = feature_stop

    return rmse_vec


def preload_model_artifacts(nn_path, use_pca, n_folds, n_ensembles):
    print("Preloading models, scalers, and PCA (if applicable)...")

    models = [[None for _ in range(n_ensembles)] for _ in range(n_folds)]
    scalers_x = []
    scalers_y = []
    pca_list = [] if use_pca else None

    for fold_idx, fold_no in enumerate(range(1, n_folds + 1)):
        scalers_x.append(joblib.load(os.path.join(nn_path, f"scaler_x_fold{fold_no}.pkl")))
        scalers_y.append(joblib.load(os.path.join(nn_path, f"scaler_y_fold{fold_no}.pkl")))
        if use_pca:
            pca_list.append(joblib.load(os.path.join(nn_path, f"pca_y_fold{fold_no}.pkl")))

        for ens_no in range(1, n_ensembles + 1):
            model_path = os.path.join(nn_path, f"model_fold{fold_no}_ens{ens_no}.h5")
            model = load_model_with_noise_support(model_path)
            models[fold_idx][ens_no - 1] = model

    print("Preloading complete.\n")
    return models, scalers_x, scalers_y, pca_list


def print_noise_summary(noise):
    print(f"Noise mean (first 5): {np.mean(noise, axis=0)[:5]}")
    print(f"Noise mean (last 5): {np.mean(noise, axis=0)[-5:]}")
    print(f"Noise std (first 5): {np.std(noise, axis=0)[:5]}")
    print(f"Noise std (last 5): {np.std(noise, axis=0)[-5:]}")


def run_monte_carlo_predictions(
    nn_path,
    loaded_data,
    run_config,
    use_pca,
    n_folds,
    n_ensembles,
    p_noise,
    ssh_noise,
    n_mc,
):
    """Run Monte Carlo predictions with featurewise noise."""
    n_samples, n_features = loaded_data.input_all.shape
    n_outputs = loaded_data.psi.shape[1]

    rmse_vec = build_feature_noise_vector(
        run_config.covariate_names,
        loaded_data.input_num_ind,
        n_features,
        p_noise,
        ssh_noise,
    )

    mean_mc = np.zeros((n_samples, n_outputs))
    m2_mc = np.zeros((n_samples, n_outputs))
    mean_mc_lpf = np.zeros((n_samples, n_outputs))
    m2_mc_lpf = np.zeros((n_samples, n_outputs))
    r2_each_mc = np.empty((n_mc, n_outputs))

    models, scalers_x, scalers_y, pca_list = preload_model_artifacts(nn_path, use_pca, n_folds, n_ensembles)

    for mc_idx in range(n_mc):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n=== Iteration {mc_idx + 1}/{n_mc} | Time: {timestamp} ===")

        noise = np.random.normal(loc=0.0, scale=rmse_vec, size=loaded_data.input_all.shape)
        print_noise_summary(noise)
        input_noisy = loaded_data.input_all + noise

        pred_folds = np.zeros((n_folds, n_samples, n_outputs))
        for fold_idx in range(n_folds):
            x_scaled = scalers_x[fold_idx].transform(input_noisy)
            preds_ensemble = np.empty((n_ensembles, n_samples, n_outputs))

            for ens_idx, model in enumerate(models[fold_idx]):
                y_pred = model.predict(x_scaled, verbose=0)
                if use_pca:
                    y_pred = pca_list[fold_idx].inverse_transform(y_pred)
                preds_ensemble[ens_idx] = y_pred

            pred_mean_fold = preds_ensemble.mean(axis=0)
            pred_folds[fold_idx] = scalers_y[fold_idx].inverse_transform(pred_mean_fold)

        pred_mean_mc = pred_folds.mean(axis=0)
        pred_mean_mc_lpf = apply_filter_by_realization(pred_mean_mc)

        for output_idx in range(n_outputs):
            r2_each_mc[mc_idx, output_idx] = r2_score(loaded_data.psi[:, output_idx], pred_mean_mc[:, output_idx])

        delta = pred_mean_mc - mean_mc
        mean_mc += delta / (mc_idx + 1)
        delta2 = pred_mean_mc - mean_mc
        m2_mc += delta * delta2

        delta_lpf = pred_mean_mc_lpf - mean_mc_lpf
        mean_mc_lpf += delta_lpf / (mc_idx + 1)
        delta2_lpf = pred_mean_mc_lpf - mean_mc_lpf
        m2_mc_lpf += delta_lpf * delta2_lpf

    var_mc = m2_mc / (n_mc - 1) if n_mc > 1 else np.zeros_like(m2_mc)
    var_mc_lpf = m2_mc_lpf / (n_mc - 1) if n_mc > 1 else np.zeros_like(m2_mc_lpf)

    return MonteCarloOutput(
        pred_mean=mean_mc,
        pred_mean_yz=reshape_masked_field(mean_mc, loaded_data.psi_mask, loaded_data.n_levels, loaded_data.n_lats),
        pred_variance_yz=reshape_masked_field(var_mc, loaded_data.psi_mask, loaded_data.n_levels, loaded_data.n_lats),
        pred_mean_lpf=mean_mc_lpf,
        pred_mean_yz_lpf=reshape_masked_field(
            mean_mc_lpf, loaded_data.psi_mask, loaded_data.n_levels, loaded_data.n_lats
        ),
        pred_variance_yz_lpf=reshape_masked_field(
            var_mc_lpf, loaded_data.psi_mask, loaded_data.n_levels, loaded_data.n_lats
        ),
        r2_each_mc=r2_each_mc,
    )


def compute_r2_fields(loaded_data, pred_mean, pred_mean_lpf):
    """Compute pointwise R2 before and after the post-training low-pass filter."""
    psi_lpf = apply_filter_by_realization(loaded_data.psi)

    n_outputs = loaded_data.psi.shape[1]
    r2_mean = np.empty(n_outputs)
    r2_mean_lpf = np.empty(n_outputs)

    for output_idx in range(n_outputs):
        r2_mean[output_idx] = r2_score(loaded_data.psi[:, output_idx], pred_mean[:, output_idx])
        r2_mean_lpf[output_idx] = r2_score(psi_lpf[:, output_idx], pred_mean_lpf[:, output_idx])

    return (
        reshape_masked_vector(r2_mean, loaded_data.psi_mask, loaded_data.n_levels, loaded_data.n_lats),
        reshape_masked_vector(r2_mean_lpf, loaded_data.psi_mask, loaded_data.n_levels, loaded_data.n_lats),
    )


# ========== Output Helpers ==========
def build_result_suffix(p_noise, ssh_noise, n_mc):
    suffix_parts = [f"PNoise{p_noise:g}"]
    if ssh_noise != 0:
        suffix_parts.append(f"SSHNoise{int(round(ssh_noise * 100))}")
    suffix_parts.append(f"MC{n_mc}")
    return "_".join(suffix_parts)


def plot_r2(lat_psi, rho2, r2_mean_yz, r2_mean_yz_lpf, savepath, data_str, result_suffix):
    """Plot and save R2 fields."""
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator

    def save_single_plot(field, title, filename):
        fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
        pcm = ax.pcolormesh(lat_psi, rho2, field, cmap="gist_rainbow_r", shading="auto", vmin=0, vmax=1)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("Latitude")
        ax.set_ylabel("Potential density")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        cbar = fig.colorbar(pcm, ax=ax, orientation="vertical")
        cbar.set_label(r"$R^2$")

        fig.savefig(os.path.join(savepath, filename), dpi=300)
        plt.show()
        plt.close(fig)

    save_single_plot(
        r2_mean_yz,
        r"$R^2$ computed using the mean prediction across all NNs",
        f"TestR2_{data_str}_{result_suffix}.png",
    )
    save_single_plot(
        r2_mean_yz_lpf,
        r"$R^2$ computed using the mean prediction across all NNs; 10-year LPF after training",
        f"TestR2_{data_str}_{result_suffix}_10YLPFafter.png",
    )


def save_results(savepath, data_str, result_suffix, loaded_data, predictions, r2_mean_yz, r2_mean_yz_lpf):
    variables_dict = {
        "r2_mean_yz": r2_mean_yz,
        "r2_EachMC": predictions.r2_each_mc,
        "pred_mean_yz": predictions.pred_mean_yz,
        "pred_variance_yz": predictions.pred_variance_yz,
        "Psi": loaded_data.psi,
        "r2_mean_yz_LPFafter": r2_mean_yz_lpf,
        "pred_mean_yz_LPF": predictions.pred_mean_yz_lpf,
        "pred_variance_yz_LPF": predictions.pred_variance_yz_lpf,
        "Noise_interNum": NOISE_INTER_NUM,
        "rho2": loaded_data.rho2,
        "lat_psi": loaded_data.lat_psi,
    }
    sio.savemat(os.path.join(savepath, f"R2_{data_str}_{result_suffix}.mat"), variables_dict)


# ========== Main Workflow ==========
def process_cmip_case(cmip_name, data_str, use_pca, nn_path, savepath, run_config):
    print(f"\n=== Processing {cmip_name} ({data_str}) ===")

    data_dir = resolve_data_dir(cmip_name, data_str)
    loaded_data = load_data(
        data_dir=data_dir,
        str_file=STR_FILE,
        run_config=run_config,
        lpf_data_suffix=LPF_DATA_SUFFIX,
        amoc26_input=AMOC26_INPUT,
        amoc56_input=AMOC56_INPUT,
    )

    predictions = run_monte_carlo_predictions(
        nn_path=nn_path,
        loaded_data=loaded_data,
        run_config=run_config,
        use_pca=use_pca,
        n_folds=N_FOLDS,
        n_ensembles=N_ENSEMBLES,
        p_noise=P_NOISE,
        ssh_noise=SSH_NOISE,
        n_mc=NOISE_INTER_NUM,
    )

    r2_mean_yz, r2_mean_yz_lpf = compute_r2_fields(
        loaded_data,
        predictions.pred_mean,
        predictions.pred_mean_lpf,
    )

    result_suffix = build_result_suffix(P_NOISE, SSH_NOISE, NOISE_INTER_NUM)
    save_results(savepath, data_str, result_suffix, loaded_data, predictions, r2_mean_yz, r2_mean_yz_lpf)
    plot_r2(loaded_data.lat_psi, loaded_data.rho2, r2_mean_yz, r2_mean_yz_lpf, savepath, data_str, result_suffix)


def build_nn_path(model_base_path, nn_name, input_var):
    return os.path.join(model_base_path, MOC_STR, f"results{LPF_STR}", nn_name, input_var)


def main():
    require_existing_directory(CMIP_DATA_ROOT, "CMIP data root")

    for covariate_names in COVARIATE_NAMES_LIST:
        run_config = prepare_run_config(covariate_names, AMOC26_INPUT, AMOC56_INPUT)
        print(f"\n=== Testing covariates: {run_config.covariate_names} ===")
        if run_config.noise_size:
            print(f"NoiseSize = {run_config.noise_size}")

        for nnstr_base in NNSTR_LIST:
            nn_name = f"{nnstr_base}{LPF_STR}"
            use_pca = "PCA" in nn_name
            print(f"\n=== Testing NNstr: {nn_name} ===")

            for model_base_path in MODEL_BASE_PATHS:
                require_existing_directory(model_base_path, "Model output root")
                nn_path = build_nn_path(model_base_path, nn_name, run_config.input_var)
                if not os.path.isdir(nn_path):
                    print(f"Skipping missing NN path: {nn_path}")
                    continue

                savepath = os.path.join(nn_path, "MonteCarlo")
                os.makedirs(savepath, exist_ok=True)

                for cmip_name, data_str in CMIP_CASES:
                    process_cmip_case(cmip_name, data_str, use_pca, nn_path, savepath, run_config)


if __name__ == "__main__":
    main()
