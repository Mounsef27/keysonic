"""Functions for retrieving and saving KeySonic matrix data."""

from pathlib import Path

import numpy as np

from keysonic.utils.utils import get_matrix, read_file


def retrieve_data(
    base_dir: str = "../Multi-Keyboard Acoustic (MKA) Datasets/MKA datasets/All Dataset",
    folder_name: str = "preprocessed_data",
    raw_data_folder: str = "Sound Segment(.matrix)",
) -> None:
    """Extract and save keyboard acoustic matrices as NumPy files.

    Processes all files in each key subdirectory, regardless of extension.
    Files that cannot be decoded as UTF-8 are skipped.
    Existing output files are overwritten.

    Args:
        base_dir: Path to the directory containing the dataset.
        folder_name: Name of the output folder created inside base_dir.
        raw_data_folder: Name of the folder containing key subdirectories.

    Returns:
        None. Extracted matrices are saved as .npy files.

    Raises:
        FileNotFoundError: If the raw data folder does not exist.
        OSError: If a directory or file cannot be accessed or written.
        ValueError: If matrix dimensions or numerical values are invalid.
        IndexError: If a matrix entry index is outside its dimensions.
    """
    base_path = Path(base_dir).expanduser().resolve()
    raw_dir = base_path / raw_data_folder
    output_dir = base_path / folder_name

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {raw_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Source directory: {raw_dir}")

    for directory in sorted(raw_dir.iterdir()):
        if not directory.is_dir():
            continue

        key_output_dir = output_dir / directory.name
        key_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nKey: {directory.name}")

        for file in sorted(directory.iterdir()):
            if not file.is_file():
                continue

            try:
                data = read_file(file)
            except UnicodeDecodeError:
                print(f"Skipped: {file.name} — cannot decode as UTF-8")
                continue

            matrix = get_matrix(data)
            output_path = key_output_dir / f"{file.name}.npy"

            np.save(output_path, matrix)

            print(f"  File: {file.name} | Shape: {matrix.shape}")
            print(f"  Saved: {output_path}")