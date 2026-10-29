"""Utilities for building training and validation data loaders."""

import torch
from torch.utils.data import DataLoader, Dataset


def build_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    *,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders.

    Args:
        train_dataset: Dataset containing training samples.
        val_dataset: Dataset containing validation samples.
        batch_size: Maximum number of samples per batch.
        num_workers: Number of subprocesses used to load data.
            Zero loads data in the main process.
        pin_memory: Whether to use pinned memory for GPU transfers.
        seed: Seed controlling the training sample order.

    Returns:
        A tuple containing the training and validation data loaders.

    Raises:
        ValueError: If batch_size is less than one, num_workers is
            negative, or either dataset is empty.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be greater than zero.")

    if num_workers < 0:
        raise ValueError("num_workers must be zero or greater.")

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Training and validation datasets must not be empty.")

    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader