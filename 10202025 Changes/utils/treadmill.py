"""Treadmill data processing utilities for ground reaction forces.

This module provides functionality to load, process, and normalize vertical ground
reaction force (vGRF) data from treadmill measurements. It handles data from both
left and right force plates and provides methods for resampling and normalization.
"""

from typing import Tuple
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

GRAVITY = 9.81
@dataclass
class TreadmillData:
    """Represents processed treadmill force data.
    DataFrame columns are time and vgrf.
    
    Attributes:
        left_df: Processed vGRF data from left force plate
        right_df: Processed vGRF data from right force plate
    """
    left_df: pd.DataFrame
    right_df: pd.DataFrame

def _resample_vgrf(df: pd.DataFrame, resample_freq: int) -> pd.DataFrame:
    """Resample vGRF data to a specified frequency.
    
    Args:
        df: DataFrame containing 'time' and 'vgrf' columns
        resample_freq: Target sampling frequency in Hz
        
    Returns:
        DataFrame with resampled vGRF data
    """
    start, end = df['time'].iloc[0], df['time'].iloc[-1]
    interval = 1 / resample_freq

    out_df = pd.DataFrame({
        'time': np.arange(start, end, interval),
        'vgrf': np.interp(
            np.arange(start, end, interval),
            df['time'],
            df['vgrf']
        )
    })
    return out_df

def _read_treadmill_mot(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read and parse treadmill data from MOT file.
    
    Args:
        file_path: Path to the MOT file containing treadmill data
        
    Returns:
        Tuple of DataFrames (left, right) containing raw vGRF data
        
    Raises:
        FileNotFoundError: If the specified file doesn't exist
    """
    def organize(df: pd.DataFrame) -> pd.DataFrame:
        """Sort and clean treadmill DataFrame."""
        return df.sort_values(by='time').reset_index(drop=True)

    with open(file_path, 'r', encoding='utf-8') as fp:
        # Skip metadata header
        for _ in range(6):
            fp.readline()

        df = pd.read_csv(fp, sep='\\s+')

        # Extract force data for each plate
        force_data = {
            # TODO: Are these flipped?
            'left': pd.DataFrame({
                'time': df['time'],
                'vgrf': df.apply(
                    lambda x: (x['1_ground_force_vx'], x['1_ground_force_vy']),
                    axis=1
                )
            }),
            'right': pd.DataFrame({
                'time': df['time'],
                'vgrf': df.apply(
                    lambda x: (x['ground_force_vx'], x['ground_force_vy']),
                    axis=1
                )
            })
        }

        return tuple(organize(df) for df in force_data.values())

def _process_treadmill_df(
    df: pd.DataFrame,
    subject_weight: float,
    resample_freq: int
) -> pd.DataFrame:
    """Process treadmill data with normalization and resampling.
    
    Args:
        df: Raw treadmill DataFrame
        subject_weight: Subject's weight in kg
        resample_freq: Target sampling frequency in Hz
        
    Returns:
        Processed DataFrame with normalized and resampled vGRF data
    """
    # Normalize force vectors and convert to body weights
    df['vgrf'] = df['vgrf'].apply(np.linalg.norm)
    df['vgrf'] /= (subject_weight * GRAVITY)

    # Resample data
    return _resample_vgrf(df, resample_freq)

def load_treadmill_data(
    file_path: str,
    subject_weight: float,
    resample_freq: int = 100
) -> TreadmillData:
    """Load and process treadmill data from a MOT file.
    
    Args:
        file_path: Path to the MOT file containing treadmill data
        subject_weight: Subject's weight in kg
        resample_freq: Target sampling frequency in Hz
        
    Returns:
        TreadmillData object containing processed data from both force plates
        
    Raises:
        FileNotFoundError: If the specified file doesn't exist
        ValueError: If subject_weight or resample_freq is not positive
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if subject_weight <= 0:
        raise ValueError(f"subject_weight must be positive, got {subject_weight}")
    if resample_freq <= 0:
        raise ValueError(f"resample_freq must be positive, got {resample_freq}")

    # Load and process treadmill data
    unprocessed_data = _read_treadmill_mot(file_path)
    processed_data = [
        _process_treadmill_df(df, subject_weight, resample_freq)
        for df in unprocessed_data
    ]

    return TreadmillData(*processed_data)
