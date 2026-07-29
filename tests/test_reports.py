"""
tests/test_reports.py – Reports directory integrity checks.

Verifies that the reports/ directory contains the expected evidence files
and that they contain meaningful content (not empty stubs).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

REPORTS_DIR = Path(__file__).parents[1] / "reports" / "v80_attention"


class TestReportsDirectory:
    """Reports directory must exist and contain required files."""

    def test_v80_attention_dir_exists(self):
        assert REPORTS_DIR.exists(), (
            "reports/v80_attention/ must exist as evidence of synthesis results"
        )

    def test_configuration_md_exists(self):
        assert (REPORTS_DIR / "configuration.md").exists()

    def test_configuration_md_mentions_v80(self):
        content = (REPORTS_DIR / "configuration.md").read_text(encoding="utf-8")
        assert "V80" in content or "v80" in content.lower()

    def test_configuration_md_mentions_latency(self):
        """configuration.md must record the claimed latency value."""
        content = (REPORTS_DIR / "configuration.md").read_text(encoding="utf-8")
        assert "12.16" in content or "12.1" in content, (
            "configuration.md must document the 12.16 ms synthesis estimate"
        )

    def test_configuration_md_clarifies_synthesis_estimate(self):
        """Must clearly state these are synthesis estimates, not board results."""
        content = (REPORTS_DIR / "configuration.md").read_text(encoding="utf-8")
        assert "Synthesis" in content or "Estimate" in content or "synthesis" in content.lower()

    def test_command_txt_exists(self):
        assert (REPORTS_DIR / "command.txt").exists(), (
            "command.txt must record the exact command used to generate results"
        )

    def test_command_txt_references_onnx(self):
        content = (REPORTS_DIR / "command.txt").read_text(encoding="utf-8")
        assert "onnx" in content.lower() or "main.py" in content

    def test_readme_exists(self):
        assert (REPORTS_DIR / "README.md").exists(), (
            "reports/v80_attention/README.md must explain the directory contents"
        )

    def test_golden_compare_py_exists(self):
        assert (REPORTS_DIR / "golden_compare.py").exists(), (
            "golden_compare.py (even as a stub) must be present"
        )

    def test_utilization_rpt_exists(self):
        assert (REPORTS_DIR / "utilization.rpt").exists(), (
            "utilization.rpt must contain extracted resource utilization data"
        )

    def test_utilization_rpt_not_empty(self):
        content = (REPORTS_DIR / "utilization.rpt").read_text(encoding="utf-8")
        assert len(content.strip()) > 50, (
            "utilization.rpt must not be empty or trivial"
        )
