"""
tests/test_export_to_slash.py – Tests for the SLASH export adapter.

All tests run without hardware. ExportError is raised for invalid inputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from integrations.slash.export_to_slash import SlashExporter, ExportError


class TestPreflightChecks:
    """SlashExporter must fail fast with specific errors for invalid inputs."""

    def test_missing_hls_project_raises(self, tmp_path):
        exporter = SlashExporter(
            hls_project=tmp_path / "nonexistent_project",
            project_name="test",
            output_dir=tmp_path / "out",
        )
        with pytest.raises(ExportError, match="not found"):
            exporter.export()

    def test_missing_kernel_source_raises(self, tmp_path):
        """Project has no output.cpp or slrX.cpp → should raise."""
        hls = tmp_path / "hls_proj"
        src = hls / "src"
        src.mkdir(parents=True)
        (src / "output.h").write_text("#pragma once\n", encoding="utf-8")
        (hls / "nlp.log").write_text("TC0_0 = 197\n", encoding="utf-8")

        exporter = SlashExporter(
            hls_project=hls,
            project_name="test",
            output_dir=tmp_path / "out",
        )
        with pytest.raises(ExportError, match="No HLS kernel source"):
            exporter.export()

    def test_missing_nlp_log_raises(self, tmp_path):
        """NLP log missing → should raise during preflight."""
        hls = tmp_path / "hls_proj"
        src = hls / "src"
        src.mkdir(parents=True)
        (src / "output.cpp").write_text("void kernel_nlp(float* A) {}", encoding="utf-8")
        (src / "output.h").write_text("#pragma once\n", encoding="utf-8")
        # No nlp.log

        exporter = SlashExporter(
            hls_project=hls,
            project_name="test",
            output_dir=tmp_path / "out",
        )
        with pytest.raises(ExportError, match="nlp.log"):
            exporter.export()

    def test_missing_slash_root_raises_when_provided(self, tmp_path, tmp_hls_project):
        """Non-existent slash_root must raise."""
        exporter = SlashExporter(
            hls_project=tmp_hls_project,
            slash_root=tmp_path / "nonexistent_SLASH",
            project_name="test",
            output_dir=tmp_path / "out",
        )
        with pytest.raises(ExportError, match="SLASH root directory not found"):
            exporter.export()

    def test_missing_abstract_shell_dcp_raises(self, tmp_path, tmp_hls_project):
        """SLASH root present but no DCP → should warn/raise."""
        slash = tmp_path / "SLASH"
        slash.mkdir()  # exists but no DCP inside

        exporter = SlashExporter(
            hls_project=tmp_hls_project,
            slash_root=slash,
            project_name="test",
            output_dir=tmp_path / "out",
        )
        with pytest.raises(ExportError, match="abstract shell DCP"):
            exporter.export()

    def test_reserved_word_kernel_name_raises(self, tmp_path):
        """Kernel named 'module' must be rejected."""
        hls = tmp_path / "hls_proj"
        src = hls / "src"
        src.mkdir(parents=True)
        # Use a Verilog reserved word as the function name
        (src / "output.cpp").write_text("void module(float* A) {}", encoding="utf-8")
        (src / "output.h").write_text("#pragma once\n", encoding="utf-8")
        (hls / "nlp.log").write_text("TC0_0 = 197\n", encoding="utf-8")

        exporter = SlashExporter(
            hls_project=hls,
            project_name="test",
            output_dir=tmp_path / "out",
        )
        with pytest.raises(ExportError, match="reserved word"):
            exporter.export()


class TestDryRun:
    """Dry-run mode must pass all checks but write NO files."""

    def test_dry_run_no_files_written(self, tmp_path, tmp_hls_project, tmp_slash_root):
        out = tmp_path / "output"
        exporter = SlashExporter(
            hls_project=tmp_hls_project,
            slash_root=tmp_slash_root,
            project_name="dry_test",
            output_dir=out,
            dry_run=True,
            verbose=False,
        )
        exporter.export()
        # Output directory should NOT be created
        assert not out.exists(), "Dry-run must not create output directory"

    def test_dry_run_validates_project(self, tmp_path):
        """Dry-run must still raise on invalid project (not skip checks)."""
        exporter = SlashExporter(
            hls_project=tmp_path / "does_not_exist",
            project_name="test",
            output_dir=tmp_path / "out",
            dry_run=True,
        )
        with pytest.raises(ExportError, match="not found"):
            exporter.export()


class TestExportBundle:
    """A successful export must produce all required files."""

    REQUIRED_FILES = {
        "output.cfg",
        "config.cfg",
        "slr_constraints.tcl",
        "host.cpp",
        "CMakeLists.txt",
        "run_v80.sh",
        "autohls_flow_manifest.json",
    }

    def test_export_produces_required_files(
        self, tmp_path, tmp_hls_project, tmp_slash_root
    ):
        out = tmp_path / "slash_out"
        exporter = SlashExporter(
            hls_project=tmp_hls_project,
            slash_root=tmp_slash_root,
            project_name="test_project",
            output_dir=out,
            dry_run=False,
            verbose=False,
        )
        exporter.export()

        produced = {f.name for f in out.rglob("*") if f.is_file()}
        for required in self.REQUIRED_FILES:
            assert required in produced, f"Missing required file: {required}"

    def test_run_v80_is_executable(self, tmp_path, tmp_hls_project, tmp_slash_root):
        out = tmp_path / "slash_out2"
        exporter = SlashExporter(
            hls_project=tmp_hls_project,
            slash_root=tmp_slash_root,
            project_name="test_project",
            output_dir=out,
            dry_run=False,
            verbose=False,
        )
        exporter.export()
        run_sh = out / "run_v80.sh"
        assert run_sh.exists()
        assert run_sh.stat().st_mode & 0o111, "run_v80.sh must be executable"

    def test_manifest_contains_required_keys(
        self, tmp_path, tmp_hls_project, tmp_slash_root
    ):
        out = tmp_path / "slash_out3"
        exporter = SlashExporter(
            hls_project=tmp_hls_project,
            slash_root=tmp_slash_root,
            project_name="test_project",
            output_dir=out,
            dry_run=False,
            verbose=False,
        )
        exporter.export()

        manifest = json.loads((out / "autohls_flow_manifest.json").read_text(encoding="utf-8"))
        required_keys = {
            "schema_version", "generation_timestamp", "autohls_flow_version",
            "project_name", "kernel_name", "kernel_count", "target_freq_mhz",
            "hls_part", "slr_mapping", "nlp_tiling", "synthesis_status",
        }
        for key in required_keys:
            assert key in manifest, f"Manifest missing key: {key}"

    def test_manifest_nlp_tiling_extracted(
        self, tmp_path, tmp_hls_project, tmp_slash_root
    ):
        """NLP tiling factors from nlp.log must appear in the manifest."""
        out = tmp_path / "slash_out4"
        exporter = SlashExporter(
            hls_project=tmp_hls_project,
            slash_root=tmp_slash_root,
            project_name="test_project",
            output_dir=out,
            dry_run=False,
            verbose=False,
        )
        exporter.export()

        manifest = json.loads((out / "autohls_flow_manifest.json").read_text(encoding="utf-8"))
        tiling = manifest.get("nlp_tiling", {})
        assert "TC0_0" in tiling, "TC0_0 tiling factor must be in manifest"
        assert tiling["TC0_0"] == 197

    def test_output_cfg_has_vivado_flow(
        self, tmp_path, tmp_hls_project, tmp_slash_root
    ):
        out = tmp_path / "slash_out5"
        exporter = SlashExporter(
            hls_project=tmp_hls_project,
            slash_root=tmp_slash_root,
            project_name="test_project",
            output_dir=out,
            dry_run=False,
            verbose=False,
        )
        exporter.export()
        cfg = (out / "output.cfg").read_text(encoding="utf-8")
        assert "flow_target=vivado" in cfg, "output.cfg must set flow_target=vivado"

    def test_output_cfg_no_u55c(self, tmp_path, tmp_hls_project, tmp_slash_root):
        """output.cfg must NOT contain U55C part string."""
        out = tmp_path / "slash_out6"
        exporter = SlashExporter(
            hls_project=tmp_hls_project,
            slash_root=tmp_slash_root,
            project_name="test_project",
            output_dir=out,
            dry_run=False,
            verbose=False,
        )
        exporter.export()
        cfg = (out / "output.cfg").read_text(encoding="utf-8")
        assert "xcu55c" not in cfg, "output.cfg must not contain xcu55c"

    def test_force_overwrites_existing(
        self, tmp_path, tmp_hls_project, tmp_slash_root
    ):
        out = tmp_path / "slash_out7"
        exporter = SlashExporter(
            hls_project=tmp_hls_project,
            slash_root=tmp_slash_root,
            project_name="test_project",
            output_dir=out,
            dry_run=False,
            verbose=False,
            force=True,
        )
        exporter.export()
        # Second export with force must succeed (not raise)
        exporter.export()
        assert (out / "autohls_flow_manifest.json").exists()

    def test_no_force_existing_dir_raises(
        self, tmp_path, tmp_hls_project, tmp_slash_root
    ):
        out = tmp_path / "slash_out8"
        exporter = SlashExporter(
            hls_project=tmp_hls_project,
            slash_root=tmp_slash_root,
            project_name="test_project",
            output_dir=out,
            dry_run=False,
            verbose=False,
            force=False,
        )
        exporter.export()
        # Second export without force must raise
        with pytest.raises(ExportError, match="already exists"):
            exporter.export()
