"""Utilities for formatting and saving stance analysis results."""

from typing import List
import pandas as pd

def _get_stance_peak(stance: pd.Series) -> float:
    """Returns the maximum value in the first 30 frames of a stance."""
    if stance.shape[0] < 30:
        raise ValueError(f"Stance is too short: {stance.shape[0]}")
    return stance[:30].max()

def make_peak_results(left_stances: List[pd.Series],
                      right_stances: List[pd.Series]
                      ) -> pd.DataFrame:
    """Creates a DataFrame of peak values for left and right stances.
    
    Returns:
        DataFrame with columns: step_id, side, peak_value
    """
    all_stances = left_stances + right_stances
    return pd.DataFrame({
        'step_id': list(range(len(left_stances))) + list(range(len(right_stances))),
        'side': ['left'] * len(left_stances) + ['right'] * len(right_stances),
        'peak_value': [_get_stance_peak(stance) for stance in all_stances]
    })

def make_stance_results(left_stances: List[pd.Series],
                        right_stances: List[pd.Series]
                        ) -> pd.DataFrame:
    """Creates a DataFrame of stance values over time.
    
    Returns:
        DataFrame with columns: frame, left_stance_0, left_stance_1, ..., right_stance_0, ...
        Each row represents a frame, with values for each stance at that frame.
    """
    columns = ['frame']
    for i in range(len(left_stances)):
        columns.append(f'left_stance_{i}')
    for i in range(len(right_stances)):
        columns.append(f'right_stance_{i}')

    rows = []
    for j in range(100):
        new_row = [j+1] # Add sequence id
        for stance in left_stances:
            new_row.append(stance[j])
        for stance in right_stances:
            new_row.append(stance[j])
        rows.append(new_row)

    return pd.DataFrame(rows, columns=columns)
