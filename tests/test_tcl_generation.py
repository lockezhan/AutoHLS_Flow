"""
tests/test_tcl_generation.py – Tests for device-aware TCL script generation.

The critical invariant: any TCL script generated for Alveo V80 must NOT
contain the U55C part string (xcu55c-fsvh2892-2L-e).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / "code_gen"))

from code_gen.write_tcl import generate_tcl, generate_csim, generate_tcl_from_profile

# Strings that must never appear in V80-generated output
V80_FORBIDDEN = ["xcu55c", "U55C", "xilinx_u55c_gen3x16_xdma_3_202210_1"]
V80_EXPECTED_PART = "xcv80-lsva4737-2MHP-e-S"


class TestV80TclGeneration:
    """V80 TCL scripts must use V80 part and not contain any U55C strings."""

    def test_generate_tcl_default_is_v80(self, tmp_path):
        """Default generate_tcl() must produce V80 part (safe default)."""
        out = tmp_path / "vitis.tcl"
        generate_tcl(str(out))
        content = out.read_text(encoding="utf-8")
        assert V80_EXPECTED_PART in content

    def test_generate_tcl_v80_part_explicit(self, tmp_path):
        """Explicitly passing V80 part must produce correct part."""
        out = tmp_path / "vitis.tcl"
        generate_tcl(str(out), part=V80_EXPECTED_PART, freq="300")
        content = out.read_text(encoding="utf-8")
        assert V80_EXPECTED_PART in content

    def test_generate_tcl_v80_no_u55c(self, tmp_path, v80_profile):
        """TCL for V80 must not contain any U55C strings."""
        out = tmp_path / "vitis.tcl"
        generate_tcl_from_profile(str(out), v80_profile, mode="csynth")
        content = out.read_text(encoding="utf-8")
        for forbidden in V80_FORBIDDEN:
            assert forbidden not in content, (
                f"V80 TCL contains forbidden string: '{forbidden}'"
            )

    def test_generate_csim_v80_no_u55c(self, tmp_path, v80_profile):
        """csim TCL for V80 must not contain any U55C strings."""
        out = tmp_path / "csim.tcl"
        generate_tcl_from_profile(str(out), v80_profile, mode="csim")
        content = out.read_text(encoding="utf-8")
        for forbidden in V80_FORBIDDEN:
            assert forbidden not in content

    def test_generate_tcl_v80_has_300mhz(self, tmp_path, v80_profile):
        """V80 TCL must use 300 MHz (V80 default frequency)."""
        out = tmp_path / "vitis.tcl"
        generate_tcl_from_profile(str(out), v80_profile, mode="csynth")
        content = out.read_text(encoding="utf-8")
        assert "300MHz" in content

    def test_generate_tcl_u55c_has_correct_part(self, tmp_path, u55c_profile):
        """U55C TCL must contain the U55C part string."""
        out = tmp_path / "vitis.tcl"
        generate_tcl_from_profile(str(out), u55c_profile, mode="csynth")
        content = out.read_text(encoding="utf-8")
        assert "xcu55c-fsvh2892-2L-e" in content

    def test_generate_tcl_u55c_no_v80(self, tmp_path, u55c_profile):
        """U55C TCL must not contain xcv80 strings."""
        out = tmp_path / "vitis.tcl"
        generate_tcl_from_profile(str(out), u55c_profile, mode="csynth")
        content = out.read_text(encoding="utf-8")
        assert "xcv80" not in content


class TestStaticScriptTemplates:
    """The static script templates in script/ must not hardcode xcu55c."""

    def test_vitis_tcl_template_no_xcu55c(self, repo_root):
        tcl = (repo_root / "script/vitis.tcl").read_text(encoding="utf-8")
        assert "xcu55c" not in tcl, (
            "script/vitis.tcl must not hardcode xcu55c. "
            "Part is set by code_gen/write_tcl.py at runtime."
        )

    def test_csim_tcl_template_no_xcu55c(self, repo_root):
        tcl = (repo_root / "script/csim.tcl").read_text(encoding="utf-8")
        assert "xcu55c" not in tcl, (
            "script/csim.tcl must not hardcode xcu55c."
        )

    def test_hls_run_v80_redirects_to_slash(self, repo_root):
        """hls_run_v80.sh must direct users to SLASH, not the OpenCL flow."""
        sh = (repo_root / "script/hls_run_v80.sh").read_text(encoding="utf-8")
        assert "SLASH" in sh or "slash" in sh


class TestTclContent:
    """Generated TCL must contain mandatory HLS configuration directives."""

    def test_csynth_contains_csynth_design(self, tmp_path, v80_profile):
        out = tmp_path / "vitis.tcl"
        generate_tcl_from_profile(str(out), v80_profile, mode="csynth")
        content = out.read_text(encoding="utf-8")
        assert "csynth_design" in content

    def test_csim_contains_csim_design(self, tmp_path, v80_profile):
        out = tmp_path / "csim.tcl"
        generate_tcl_from_profile(str(out), v80_profile, mode="csim")
        content = out.read_text(encoding="utf-8")
        assert "csim_design" in content

    def test_csynth_contains_config_dataflow(self, tmp_path, v80_profile):
        out = tmp_path / "vitis.tcl"
        generate_tcl_from_profile(str(out), v80_profile, mode="csynth")
        content = out.read_text(encoding="utf-8")
        assert "config_dataflow" in content
