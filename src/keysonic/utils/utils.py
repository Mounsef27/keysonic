"""Utility functions for reading KeySonic matrix files."""

import re
from pathlib import Path

import numpy as np
import os
import random
from matplotlib import pyplot as plt

def read_file(path: str | Path) -> str:
    """Read the contents of a text file.

    Args:
        path: Path to the file to read.

    Returns:
        The contents of the file as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        UnicodeDecodeError: If the file cannot be decoded as UTF-8.
        OSError: If the file cannot be opened or read.
    """
    with Path(path).open(encoding="utf-8-sig") as file:
        return file.read()


def get_matrix(
    data: str,
    pattern: str = r"z\s*\[(\d+)\]\s*\[(\d+)\]\s*=\s*(\S+)",
) -> np.ndarray:
    """Extract a numerical matrix from Praat text data.

    Args:
        data: File contents containing the dimensions and matrix entries.
        pattern: Regular expression capturing the row index, column index,
            and numerical value, in that order. Indices must start at 1.

    Returns:
        A floating-point array of shape (ny, nx). Missing entries remain NaN.

    Raises:
        ValueError: If dimensions are missing or an entry is invalid.
        IndexError: If an entry index is outside the matrix dimensions.
        re.error: If the regular expression is invalid.
    """
    nx_match = re.search(r"^\s*nx\s*=\s*(\d+)", data, re.MULTILINE)
    ny_match = re.search(r"^\s*ny\s*=\s*(\d+)", data, re.MULTILINE)

    if nx_match is None or ny_match is None:
        raise ValueError("Missing matrix dimensions: nx or ny.")

    nx = int(nx_match.group(1))
    ny = int(ny_match.group(1))

    # NumPy expects the shape as (number of rows, number of columns).
    matrix = np.full((ny, nx), np.nan)

    for row, column, value in re.findall(pattern, data):
        row_index = int(row) - 1
        column_index = int(column) - 1

        # Validate indices before converting file entries to array values.
        if not (0 <= row_index < ny and 0 <= column_index < nx):
            raise IndexError(
                f"Matrix index ({row}, {column}) is outside shape ({ny}, {nx})."
            )

        matrix[row_index, column_index] = float(value)

    return matrix


def matrix_visualiser(
    data_dir: str | Path,
    figsize: tuple[int, int] = (20, 15),
) -> None:
    """Display up to 10 random matrices with their key labels.

    Args:
        data_dir: Path to the folder containing key subdirectories.
        figsize: Figure width and height in inches.

    Raises:
        FileNotFoundError: If no .npy files are found.
    """
    files = [
        file
        for file in Path(data_dir).expanduser().glob("*/*.npy")
        if file.is_file()
    ]

    if not files:
        raise FileNotFoundError(f"No .npy files found in {data_dir}")

    # Select distinct files without replacement.
    selected_files = random.sample(files, k=min(10, len(files)))

    fig, axes = plt.subplots(5, 2, figsize=figsize)

    for ax, file in zip(axes.flat, selected_files):
        matrix = np.load(file)
        label = file.parent.name

        image = ax.imshow(matrix, aspect="auto")
        ax.set_title(f"Key: {label} | File: {file.name}")
        ax.set_xlabel("Column index")
        ax.set_ylabel("Row index")
        fig.colorbar(image, ax=ax)

    for ax in list(axes.flat)[len(selected_files):]:
        ax.set_visible(False)

    fig.tight_layout()
    plt.show()