#!/usr/bin/env python3
"""
Reads ground-truth and experimental
location data for each trial under AGPS/ and Unassisted/, time-aligns
them using system-time stamps, converts to UTM, computes distance
error, averages across trials, and plots the comparison.
"""

import os
import glob
import math
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utm


# File I/O helpers 

def read_ground_truth(trial_dir):
    """
    Read Raw Data.csv and meta/time.csv from a trial directory.
    Uses positional column indexing with hardcoded names so that
    missing/blank header rows don't cause errors.
    Returns a DataFrame with an 'abs_time' column (Unix seconds).
    """
    gt_path = os.path.join(trial_dir, "Raw Data.csv")
    time_path = os.path.join(trial_dir, "meta", "time.csv")

    GT_COLUMNS = [
        "Time (s)", "Latitude (°)", "Longitude (°)", "Altitude (m)",
        "Altitude WGS84 (m)", "Speed (m/s)", "Direction (°)",
        "Distance (km)", "Horizontal Accuracy (m)", "Vertical Accuracy (m)",
        "Satellites",
    ]
    gt = pd.read_csv(gt_path, header=None, skiprows=1, names=GT_COLUMNS)


    TIME_COLUMNS = ["event", "experiment time", "system time", "system time text"]
    time_meta = pd.read_csv(time_path, header=None, skiprows=1, names=TIME_COLUMNS)

    gt_start_epoch = time_meta.iloc[0]["system time"] 

    gt["abs_time"] = gt_start_epoch + gt["Time (s)"]
    return gt, gt_start_epoch


def read_experiment(trial_dir):
    """
    Read the RSN Final *.xlsx experiment file from a trial directory.
    Uses positional column indexing with hardcoded names so that
    missing/blank header rows don't cause errors.
    Returns the Location DataFrame with an 'abs_time' column (Unix seconds).
    """
    xlsx_files = glob.glob(os.path.join(trial_dir, "RSN Final*.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError(
            f"No RSN Final *.xlsx in {trial_dir}\n"
            "  Run convert_xls.py first to convert .xls → .xlsx"
        )
    xlsx_path = xlsx_files[0]

    LOC_COLUMNS = [
        "Time (s)", "Latitude (°)", "Longitude (°)", "Height (m)",
        "Velocity (m/s)", "Direction (°)",
        "Horizontal Accuracy (m)", "Vertical Accuracy (°)",
    ]
    loc = pd.read_excel(
        xlsx_path, sheet_name="Location", engine="openpyxl",
        header=None, skiprows=1, names=LOC_COLUMNS,
    )
    
    META_TIME_COLUMNS = ["event", "experiment time", "system time", "system time text"]
    meta_time = pd.read_excel(
        xlsx_path, sheet_name="Metadata Time", engine="openpyxl",
        header=None, skiprows=1, names=META_TIME_COLUMNS,
    )

    exp_start_epoch = meta_time.iloc[0]["system time"]  # START row, col 2

    loc["abs_time"] = exp_start_epoch + loc["Time (s)"]
    return loc


# ── Core analysis ───────────────────────────────────────────────────────

def compute_error_timeseries(gt_df, exp_df, gt_start_epoch, dt=1.0, max_time=300.0):
    """
    Compute horizontal distance error (m) between experiment and ground
    truth at uniform time steps.

    Time axis is anchored to gt_start_epoch (the ground-truth START time),
    so the gap of NaNs before the experiment's first fix visually shows
    how long the TTFF took.

    Both DataFrames must have 'abs_time', 'Latitude (°)', 'Longitude (°)'.

    Returns (time_axis, error_meters, ttff) where time_axis is seconds
    since the ground-truth START, and ttff is time to first fix in seconds.
    """
    # Convert both to UTM using the utm library
    gt_utm = np.array([
        utm.from_latlon(lat, lon)[:2]
        for lat, lon in zip(gt_df["Latitude (°)"], gt_df["Longitude (°)"])
    ])
    exp_utm = np.array([
        utm.from_latlon(lat, lon)[:2]
        for lat, lon in zip(exp_df["Latitude (°)"], exp_df["Longitude (°)"])
    ])

    gt_e, gt_n = gt_utm[:, 0], gt_utm[:, 1]
    exp_e, exp_n = exp_utm[:, 0], exp_utm[:, 1]

    gt_t = gt_df["abs_time"].values
    exp_t = exp_df["abs_time"].values

    # TTFF = time from ground-truth start to experiment's first reported position
    ttff = exp_t[0] - gt_start_epoch

    # Time axis: seconds since ground-truth START
    t_axis = np.arange(0, max_time + dt, dt)
    errors = np.full_like(t_axis, np.nan)

    for i, t_rel in enumerate(t_axis):
        t_abs = gt_start_epoch + t_rel

        # No experiment data yet → leave as NaN (visible TTFF gap)
        if t_abs < exp_t[0] or t_abs > exp_t[-1]:
            continue
        if t_abs < gt_t[0] or t_abs > gt_t[-1]:
            continue

        j_exp = np.argmin(np.abs(exp_t - t_abs))
        j_gt = np.argmin(np.abs(gt_t - t_abs))

        de = exp_e[j_exp] - gt_e[j_gt]
        dn = exp_n[j_exp] - gt_n[j_gt]
        errors[i] = math.sqrt(de**2 + dn**2)

    return t_axis, errors, ttff


def collect_condition(base_dir, condition_name, dt=1.0, max_time=300.0):
    """
    Process all trials under base_dir/condition_name/Trial*.
    Returns (time_axis, list_of_error_arrays).
    """
    cond_dir = os.path.join(base_dir, condition_name)
    trial_dirs = sorted(glob.glob(os.path.join(cond_dir, "Trial*")))
    if not trial_dirs:
        raise FileNotFoundError(f"No Trial* dirs found in {cond_dir}")

    all_errors = []
    t_axis = None

    for td in trial_dirs:
        trial_name = os.path.basename(td)
        print(f"  {trial_name}: ", end="", flush=True)

        gt, gt_start_epoch = read_ground_truth(td)
        exp = read_experiment(td)
        t, err, ttff = compute_error_timeseries(
            gt, exp, gt_start_epoch, dt=dt, max_time=max_time,
        )
        t_axis = t
        all_errors.append(err)

        valid = err[~np.isnan(err)]
        if len(valid) > 0:
            print(f"TTFF = {ttff:.1f} s, "
                  f"first fix error = {valid[0]:.2f} m, "
                  f"final error = {valid[-1]:.2f} m")
        else:
            print("WARNING – no overlapping data")

    return t_axis, all_errors


# Plotting
def plot_comparison(t_axis, agps_errors, unassisted_errors, out_path="gps_error_comparison.png"):
    """Plot mean error ± 1σ for A-GPS and Unassisted conditions."""
    agps_arr = np.array(agps_errors)
    un_arr = np.array(unassisted_errors)

    # Pre-TTFF time steps are all-NaN across trials; suppress the expected warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        agps_mean = np.nanmean(agps_arr, axis=0)
        agps_std = np.nanstd(agps_arr, axis=0)
        un_mean = np.nanmean(un_arr, axis=0)
        un_std = np.nanstd(un_arr, axis=0)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(t_axis, agps_mean, color="#2196F3", linewidth=2, label="A-GPS (mean)")
    ax.fill_between(t_axis, agps_mean - agps_std, agps_mean + agps_std,
                    color="#2196F3", alpha=0.2, label="A-GPS ± 1σ")

    ax.plot(t_axis, un_mean, color="#FF5722", linewidth=2, label="Unassisted (mean)")
    ax.fill_between(t_axis, un_mean - un_std, un_mean + un_std,
                    color="#FF5722", alpha=0.2, label="Unassisted ± 1σ")

    ax.set_xlabel("Time Since Ground-Truth Collection Start (s)", fontsize=12)
    ax.set_ylabel("Horizontal Position Error (m)", fontsize=12)
    ax.set_title("Cold First Fix Localization Error: A-GPS vs Unassisted", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, t_axis[-1])
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")
    plt.show()


def main():
    # Data root = directory where this script lives
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    DT = 1.0         
    MAX_TIME = 100.0  

    print("=== A-GPS Condition ===")
    t_axis, agps_errors = collect_condition(BASE_DIR, "AGPS", dt=DT, max_time=MAX_TIME)

    print("\n=== Unassisted Condition ===")
    t_axis, unassisted_errors = collect_condition(BASE_DIR, "Unassisted", dt=DT, max_time=MAX_TIME)

    plot_comparison(t_axis, agps_errors, unassisted_errors,
                    out_path=os.path.join(BASE_DIR, "gps_error_comparison.png"))


if __name__ == "__main__":
    main()
