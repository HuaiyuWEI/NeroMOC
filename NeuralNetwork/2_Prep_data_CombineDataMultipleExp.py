"""
Combine multiple experiment archives into a single training dataset.

"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from _path_utils import ensure_directory, require_existing_directory


# ========== User settings ==========
INPUT_DIRS = [
    Path(r"E:\Data_CMIP6\ACCESS_historical\ASMOC"),
    Path(r"E:\Data_CMIP6\ACCESS_SSP585\ASMOC"),
]
OUTPUT_DIR = Path(r"E:\Data_CMIP6\ACCESS_hist+SSP585\ASMOC")

TESTDATA = False

FILE_SUFFIX = "_r36_r40" if TESTDATA else "_r1_r35"
VARIABLES_WITH_RHO = {"MOC"}
VARIABLES_TO_PROCESS = ["obp_mascon_V5", "ssh_mascon_V5", "uas_mascon_V5", "MOC"]
def concatenate_and_save(
    varname: str,
    input_dirs: list[Path],
    output_dir: Path,
    file_suffix: str,
    include_rho: bool = False,
) -> None:
    """Load one variable from several experiments, concatenate it, and save the result."""

    print(f"Processing {varname}...")
    all_temp = []
    all_temp_lpf = []
    all_realization_index = []
    mascon_lon = None
    mascon_lat = None
    rho2_full = None
    lat_psi = None

    for data_dir in input_dirs:
        file_path = data_dir / f"{varname}{file_suffix}.npz"
        if not file_path.is_file():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        print(f"  Loading from {file_path}...")
        with np.load(file_path) as data:
            all_temp.append(data[f"{varname}_ALL"])
            all_temp_lpf.append(data[f"{varname}_LPF_ALL"])
            all_realization_index.append(data["realization_index"])

            if "mascon" in varname:
                mascon_lon = data["mascon_lon"]
                mascon_lat = data["mascon_lat"]

            if include_rho:
                rho2_full = data["rho2_full"]
                lat_psi = data["lat_psi"]

    print(f"  Concatenating data for {varname}...")
    save_dict = {
        f"{varname}_ALL": np.concatenate(all_temp, axis=0),
        f"{varname}_LPF_ALL": np.concatenate(all_temp_lpf, axis=0),
        "realization_index": np.concatenate(all_realization_index, axis=0),
    }

    if "mascon" in varname:
        save_dict["mascon_lon"] = mascon_lon
        save_dict["mascon_lat"] = mascon_lat
    if include_rho:
        save_dict["rho2_full"] = rho2_full
        save_dict["lat_psi"] = lat_psi

    output_file = output_dir / f"{varname}{file_suffix}.npz"
    print(f"  Saving concatenated file to {output_file}...")
    np.savez(output_file, **save_dict)
    print(f"  Done with {varname}.\n")


def main() -> None:
    """Run the experiment concatenation workflow."""

    if not VARIABLES_TO_PROCESS:
        raise ValueError("VARIABLES_TO_PROCESS is empty; nothing to combine.")

    input_dirs = [require_existing_directory(data_dir, "Input") for data_dir in INPUT_DIRS]

    ensure_directory(OUTPUT_DIR)

    for varname in VARIABLES_TO_PROCESS:
        concatenate_and_save(
            varname=varname,
            input_dirs=input_dirs,
            output_dir=OUTPUT_DIR,
            file_suffix=FILE_SUFFIX,
            include_rho=varname in VARIABLES_WITH_RHO,
        )


if __name__ == "__main__":
    main()
