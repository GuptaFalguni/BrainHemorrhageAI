"""MONAI U-Net model factory for BrainHemorrhageAI."""

from __future__ import annotations

from typing import Any

from monai.networks.nets import UNet


def build_model(config: dict[str, Any]) -> UNet:
    """Build a configured MONAI 2D U-Net from ``config/model.yaml``.

    Args:
        config: Model configuration dictionary.

    Returns:
        Initialized MONAI ``UNet`` instance.
    """
    required_keys = (
        "spatial_dims",
        "in_channels",
        "out_channels",
        "channels",
        "strides",
    )
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise KeyError(f"Model config missing required keys: {missing}")

    return UNet(
        spatial_dims=int(config["spatial_dims"]),
        in_channels=int(config["in_channels"]),
        out_channels=int(config["out_channels"]),
        channels=tuple(int(value) for value in config["channels"]),
        strides=tuple(int(value) for value in config["strides"]),
        kernel_size=int(config.get("kernel_size", 3)),
        num_res_units=int(config.get("num_res_units", 0)),
        act=str(config.get("activation", "PRELU")),
        norm=str(config.get("normalization", "INSTANCE")),
        dropout=float(config.get("dropout", 0.0)),
    )
