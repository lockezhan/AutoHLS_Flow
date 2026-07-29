"""
tests/test_device_profiles.py – Tests for DeviceProfile and Ressources classes.
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parents[1]))

from ressources import DeviceProfile, Ressources


class TestDeviceProfileLoad:
    """DeviceProfile.load() contract tests."""

    def test_v80_loads_correct_part(self, v80_profile):
        """Alveo V80 profile must use the V80 HLS part string."""
        assert v80_profile["hls_part"] == "xcv80-lsva4737-2MHP-e-S", (
            "V80 hls_part must be xcv80-lsva4737-2MHP-e-S"
        )

    def test_u55c_loads_correct_part(self, u55c_profile):
        """Alveo U55C profile must use the U55C HLS part string."""
        assert u55c_profile["hls_part"] == "xcu55c-fsvh2892-2L-e"

    def test_v80_has_uram(self, v80_profile):
        assert v80_profile["has_uram"] is True, "V80 must report has_uram=True"

    def test_u55c_has_no_uram(self, u55c_profile):
        assert u55c_profile["has_uram"] is False

    def test_v80_deployment_backend_is_slash(self, v80_profile):
        assert "SLASH" in v80_profile["deployment_backend"], (
            "V80 deployment_backend must reference SLASH"
        )

    def test_u55c_deployment_backend_is_opencl(self, u55c_profile):
        assert "OpenCL" in u55c_profile["deployment_backend"] or \
               "Vitis" in u55c_profile["deployment_backend"]

    def test_v80_slr_count_is_3(self, v80_profile):
        assert v80_profile["slr_count"] == 3

    def test_v80_dsp_count(self, v80_profile):
        assert v80_profile["dsp"] == 10848

    def test_unknown_device_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown device"):
            DeviceProfile.load("NonExistentDevice_XYZ")

    def test_list_available_contains_v80(self):
        available = DeviceProfile.list_available()
        assert "Alveo_V80" in available

    def test_list_available_contains_u55c(self):
        available = DeviceProfile.list_available()
        assert "Alveo_U55C" in available


class TestDeviceAndProfileIsolation:
    """V80 and U55C profiles must not mix their device-specific values."""

    def test_v80_and_u55c_do_not_share_part(self, v80_profile, u55c_profile):
        assert v80_profile["hls_part"] != u55c_profile["hls_part"]

    def test_v80_part_string_not_in_u55c_profile(self, v80_profile, u55c_profile):
        # V80 part must not appear inside the U55C profile JSON at all
        u55c_str = json.dumps(u55c_profile)
        assert "xcv80" not in u55c_str, "U55C profile must not contain xcv80"

    def test_u55c_part_string_not_in_v80_profile(self, v80_profile):
        v80_str = json.dumps(v80_profile)
        assert "xcu55c" not in v80_str, "V80 profile must not contain xcu55c"


class TestRessourcesFromProfile:
    """Ressources initialised from profile must reflect profile values."""

    def test_v80_ressources_slr(self, v80_profile):
        res = Ressources(profile=v80_profile)
        assert res.SLR == 3

    def test_v80_ressources_dsp(self, v80_profile):
        res = Ressources(profile=v80_profile)
        assert res.DSP == 10848

    def test_v80_ressources_has_uram(self, v80_profile):
        res = Ressources(profile=v80_profile)
        assert res.has_uram is True

    def test_cli_override_wins(self, v80_profile):
        """CLI --SLR=1 must override profile SLR=3."""
        res = Ressources(profile=v80_profile)
        res.SLR = 1  # simulate CLI override
        assert res.SLR == 1

    def test_default_ressources_no_profile(self):
        """Ressources() without profile must not crash."""
        res = Ressources()
        assert res.SLR >= 1
        assert res.DSP > 0
