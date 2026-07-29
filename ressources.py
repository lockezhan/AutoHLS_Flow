"""
ressources.py – FPGA resource model for AutoHLS_Flow.

DeviceProfile.load() reads a device JSON from device_profiles/ and returns a dict.
Ressources() accepts an optional profile dict to initialise all hardware parameters.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

# Locate device_profiles/ relative to this file so the module works from any cwd.
_PROFILES_DIR = Path(__file__).parent / "device_profiles"


class DeviceProfile:
    """Load and validate device profile JSON files."""

    # Map the --device CLI argument to a JSON filename.
    _NAME_MAP: dict[str, str] = {
        "Alveo_V80":  "alveo_v80.json",
        "Alveo_U55C": "alveo_u55c.json",
        "AC7t1500":   "ac7t1500.json",
    }

    @classmethod
    def load(cls, name: str) -> dict:
        """Return the profile dict for the given device name.

        Parameters
        ----------
        name : str
            One of the recognised --device strings (e.g. ``"Alveo_V80"``).

        Returns
        -------
        dict
            The parsed JSON profile.

        Raises
        ------
        ValueError
            If *name* is not in the known profile map.
        FileNotFoundError
            If the JSON file is missing from device_profiles/.
        """
        if name not in cls._NAME_MAP:
            known = ", ".join(sorted(cls._NAME_MAP))
            raise ValueError(
                f"Unknown device '{name}'. Known profiles: {known}. "
                f"Add a JSON to device_profiles/ to register a new device."
            )
        json_path = _PROFILES_DIR / cls._NAME_MAP[name]
        if not json_path.exists():
            raise FileNotFoundError(
                f"Device profile '{name}' not found at expected path: {json_path}"
            )
        with open(json_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    @classmethod
    def list_available(cls) -> list[str]:
        """Return the list of known device profile names."""
        return sorted(cls._NAME_MAP.keys())


class Ressources:
    """
    Define the resources of the FPGA.

    Parameters
    ----------
    profile : dict, optional
        A device profile dict as returned by ``DeviceProfile.load()``.
        If given, hardware resource fields are initialised from the profile;
        individual CLI flags can still override them afterwards.

    Attributes
    ----------
    SLR           : int   – Number of Super Logic Regions
    DSP           : int   – Total DSP slices
    BRAM          : int   – Total BRAM_18K blocks
    ON_CHIP_MEM_SIZE : int – Total on-chip memory in bytes
    has_uram      : bool  – Whether URAM is available / should be used
    factor        : float – Resource utilisation factor (0–1)
    partitioning_max : int
    MAX_BUFFER_SIZE  : int
    MAX_UF           : int
    sizeof        : int   – Data type width in bits (32 = float32)
    """

    def __init__(self, profile: Optional[dict] = None) -> None:
        self.sizeof = 32

        # ── defaults (generic / unspecified device) ──────────────────────────
        self.SLR = 1
        self.factor = 1.0
        self.partitioning_max = 1024
        self.MAX_BUFFER_SIZE = 4096
        self.MAX_UF = 4096
        self.ON_CHIP_MEM_SIZE = 1_512_000
        self.has_uram = False

        self.DSP = int(9024 * self.factor)
        self.BRAM = int(4032 * self.factor)

        # ── override from profile ─────────────────────────────────────────────
        if profile is not None:
            self.SLR = profile.get("slr_count", self.SLR)
            self.DSP = profile.get("dsp", self.DSP)
            self.BRAM = profile.get("bram_18k", self.BRAM)
            self.ON_CHIP_MEM_SIZE = profile.get("on_chip_mem_bytes", self.ON_CHIP_MEM_SIZE)
            self.has_uram = profile.get("has_uram", self.has_uram)
            self.factor = 1.0  # profile already provides absolute values

        # ── per-SLR derived fields ────────────────────────────────────────────
        self.DSP_per_SLR = int(self.DSP / self.SLR)
        self.BRAM_per_SLR = int(self.BRAM / self.SLR)
        self.MEM_PER_SLR = int(self.ON_CHIP_MEM_SIZE / self.SLR)

        # ── operation costs (float32 only today) ──────────────────────────────
        self.DSP_per_operation: dict[str, int] = {}
        self.IL: dict[str, int] = {}
        if self.sizeof == 32:
            self.DSP_per_operation = {"+": 2, "-": 2, "*": 3, "=": 0, "/": 0}
            self.IL = {"+": 7, "*": 4, "=": 1, "-": 7, "/": 12}
        elif self.sizeof == 64:
            self.DSP_per_operation = {"+": 3, "-": 3, "*": 8, "=": 0, "/": 0}
            self.IL = {"+": 5, "*": 6, "=": 1, "-": 5, "/": 12}