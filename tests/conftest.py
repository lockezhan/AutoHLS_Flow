"""
tests/conftest.py – Shared fixtures and pytest configuration for AutoHLS_Flow tests.

All tests in this suite run WITHOUT hardware (no Vitis HLS, no V80 board).
Integration tests that require hardware are marked @pytest.mark.integration.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

# ── Root of the repository ────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parents[1].resolve()

# ── Path constants ────────────────────────────────────────────────────────────
DEVICE_PROFILES_DIR = REPO_ROOT / "device_profiles"
SCRIPT_DIR = REPO_ROOT / "script"
REPORTS_DIR = REPO_ROOT / "reports" / "v80_attention"
INTEGRATIONS_DIR = REPO_ROOT / "integrations" / "slash"


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests that require real hardware or Vitis HLS (skip in CI)."
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def v80_profile() -> dict:
    """Return the Alveo V80 device profile."""
    from ressources import DeviceProfile
    return DeviceProfile.load("Alveo_V80")


@pytest.fixture(scope="session")
def u55c_profile() -> dict:
    """Return the Alveo U55C device profile."""
    from ressources import DeviceProfile
    return DeviceProfile.load("Alveo_U55C")


@pytest.fixture
def tmp_hls_project(tmp_path: Path) -> Path:
    """Create a minimal fake HLS project directory for testing."""
    src = tmp_path / "src"
    src.mkdir()

    # Minimal kernel source with correct function signature
    (src / "output.cpp").write_text(
        "void kernel_nlp(float* A, float* B, float* C) {\n"
        "  for (int i = 0; i < 197; i++)\n"
        "    for (int j = 0; j < 768; j++)\n"
        "      C[i*768+j] = A[i*768+j] + B[i*768+j];\n"
        "}\n",
        encoding="utf-8",
    )
    (src / "output.h").write_text(
        '#pragma once\nvoid kernel_nlp(float* A, float* B, float* C);\n',
        encoding="utf-8",
    )
    (src / "slr0.cpp").write_text(
        "void kernel_nlp_slr0(float* A, float* B) {}\n", encoding="utf-8"
    )
    (src / "slr1.cpp").write_text(
        "void kernel_nlp_slr1(float* A, float* B) {}\n", encoding="utf-8"
    )
    (src / "slr2.cpp").write_text(
        "void kernel_nlp_slr2(float* A, float* B) {}\n", encoding="utf-8"
    )

    # Minimal NLP log with tiling factors
    (tmp_path / "nlp.log").write_text(
        "Objective = 12345\n"
        "TC0_0 = 197\n"
        "TC0_1 = 1\n"
        "TC1_0 = 8\n"
        "TC1_1 = 96\n"
        "TC2_0 = 192\n"
        "TC2_1 = 4\n",
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture
def tmp_slash_root(tmp_path: Path) -> Path:
    """Create a fake SLASH root with abstract shell DCP."""
    slash = tmp_path / "SLASH"
    dcp_dir = slash / "linker/resources/abstract_shell"
    dcp_dir.mkdir(parents=True)
    (dcp_dir / "abs_shell_slash.dcp").write_text("# fake DCP", encoding="utf-8")
    return slash
