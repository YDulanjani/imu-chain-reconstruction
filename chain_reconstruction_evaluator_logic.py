#Evaluator Logic for shape reconstruction from quaternions to positions.

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REFERENCE_AXIS = np.array([1.0, 0.0, 0.0])
LINK_LENGTH_CM = 5.0  # Keep consistent with the data generator.


# Rotate a vector by a quaternion similar to rando_walk_logic.py
# In here we just using with reference axis to get the direction of the next link in the chain.
def rotate_vector_by_quat(
    q: np.ndarray,
    v: np.ndarray = REFERENCE_AXIS,
) -> np.ndarray:
    w, x, y, z = q

    #seperating the vectore part of the quaternion
    qv = np.array([x, y, z])

    t = 2.0 * np.cross(qv, v)
    rotated = v + w * t + np.cross(qv, t)
    return rotated


#reconstruct the positions of the chain from the quaternions
#Current direct shape reconstruction logic 
#based on the assumption that the chain sensors are in equal length.
def quats_to_positions(
    quats: np.ndarray,
    link_length_cm: float = LINK_LENGTH_CM,
) -> np.ndarray:

    #this is for flexibility to support differnt number of sensors in the chain
    num_sensors = quats.shape[0]

    # initialize for 0
    positions = np.zeros((num_sensors, 3))
    position = np.zeros(3)

    # for each sensor extract direction & calculate position based on the link length
    for i in range(num_sensors):
        direction = rotate_vector_by_quat(
            quats[i],
            REFERENCE_AXIS,
        )

        norm = np.linalg.norm(direction)

        if norm > 0:
            # Remove small numerical errors from the rotation.
            direction = direction / norm

        position = position + direction * link_length_cm
        positions[i] = position

    return positions


def parse_cell(cell: str) -> np.ndarray:
    # Convert a "[w, x, y, z]" into an array.
    return np.array(ast.literal_eval(cell))


def sensor_columns(columns) -> list[str]:
    # Get sensor columns in sensor number order.
    cols = [c for c in columns if c.startswith("sensor_")]

    return sorted(
        cols,
        key=lambda c: int(c.split("_")[1]),
    )


def cell_row_to_array(row: pd.Series) -> np.ndarray:
    # Convert one CSV row into a sensor component array.
    sensor_cols = sensor_columns(row.index)

    return np.array([
        parse_cell(row[c])
        for c in sensor_cols
    ])


def quaternion_cell_df_to_array(
    df: pd.DataFrame,
) -> np.ndarray:
    # Convert the full DataFrame to
    # (num_samples, num_sensors, 4).
    sensor_cols = sensor_columns(df.columns)

    num_samples = len(df)
    num_sensors = len(sensor_cols)

    quats = np.zeros(
        (num_samples, num_sensors, 4)
    )

    for row_i in range(num_samples):
        quats[row_i] = cell_row_to_array(
            df.iloc[row_i]
        )

    return quats


def positions_to_cell_row(
    positions: np.ndarray,
    decimals: int = 6,
) -> dict:
    # Store each sensor position as one cell.
    row = {}

    for i in range(positions.shape[0]):
        x, y, z = positions[i]

        row[f"sensor_{i + 1}"] = (
            f"[{x:.{decimals}f}, "
            f"{y:.{decimals}f}, "
            f"{z:.{decimals}f}]"
        )

    return row


def reconstruct_positions_from_file(
    quaternion_csv_path: str | Path,
    link_length_cm: float = LINK_LENGTH_CM,
) -> pd.DataFrame:
    #read the quaternions file
    df = pd.read_csv(quaternion_csv_path)

    #convert the quaternions in the dataframe to a numpy array
    quats = quaternion_cell_df_to_array(df)

    #Reconstruct positions for each sample.
    rows = [
        positions_to_cell_row(
            quats_to_positions(
                sample_quats,
                link_length_cm,
            )
        )
        for sample_quats in quats
    ]

    out_df = pd.DataFrame(rows)

    if "sample" in df.columns:
        out_df.insert(
            0,
            "sample",
            df["sample"].values,
        )
    else:
        out_df.insert(
            0,
            "sample",
            range(1, len(out_df) + 1),
        )

    return out_df


def chain_error_cm(
    ground_truth_positions: np.ndarray,
    other_positions: np.ndarray,
) -> np.ndarray:
    # Calculate the position error for each sensor.
    #uclidean distance between the ground truth and other positions for each sensor.
    return np.linalg.norm(
        ground_truth_positions - other_positions,
        axis=1,
    )


if __name__ == "__main__":
    # Run from terminal:
    # python chain_reconstruction_evaluator_logic.py <quaternions.csv> [output.csv]
    

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    result_df = reconstruct_positions_from_file(input_path)

    # print(
    #     f"Reconstructed {len(result_df)} sample(s) "
    #     f"of X, Y, Z positions from {input_path}\n"
    # )

    print(result_df.to_string(index=False))

    if output_path:
        result_df.to_csv(output_path, index=False)
        print(f"\nWrote reconstructed positions to {output_path}")