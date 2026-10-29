"""this file contain code to create Custom dataset"""
from pathlib import Path
from collections.abc import Callable #thing that we can call it as a function

import numpy as np
import torch
from torch.utils.data import Dataset


class KeyboardDataset(Dataset):
    """Load acoustic matrices labeled by their parent directories.

    Args:
        root_dir: Path to the folder containing one subdirectory per key.
        transform: Optional transformation applied to each input tensor.

    Raises:
        FileNotFoundError: If root_dir does not exist.
        ValueError: If no NumPy files are found.
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: Callable | None = None,
    ) -> None:
        self.root_dir = Path(root_dir).expanduser()
        self.transform = transform

        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {self.root_dir}")

        # Use a stable order to assign class indices.
        self.classes = sorted(
            directory.name
            for directory in self.root_dir.iterdir()
            if directory.is_dir()
        )

        self.class_to_idx = {
            label: index
            for index, label in enumerate(self.classes)
        }

        # Store file paths and labels; load matrices only when requested.
        self.samples = [
            (file, self.class_to_idx[label])
            for label in self.classes
            for file in sorted((self.root_dir / label).glob("*.npy"))
            if file.is_file()
        ]

        if not self.samples:
            raise ValueError(f"No .npy files found in {self.root_dir}")

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load a matrix and its class index.

        Args:
            index: Position of the sample to retrieve.

        Returns:
            A float32 input tensor of shape (1, rows, columns) and
            a scalar int64 label tensor.

        Raises:
            ValueError: If the matrix is not two-dimensional.
        """
        file_path, label = self.samples[index]

        matrix = np.load(file_path, allow_pickle=False)

        if matrix.ndim != 2:
            raise ValueError(
                f"Expected a 2D matrix in {file_path}, got {matrix.shape}."
            )

        inputs = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)

        if self.transform is not None:
            inputs = self.transform(inputs)

        target = torch.tensor(label, dtype=torch.long)

        return inputs, target

        