"""
tests/test_cli.py – Tests ensuring CLI arguments match the README and documentation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from main import build_parser


class TestArgparserStructure:
    """Every flag documented in the README must exist in the parser."""

    @pytest.fixture(autouse=True)
    def parser(self) -> argparse.ArgumentParser:
        self._parser = build_parser()
        return self._parser

    def _has_arg(self, *names) -> bool:
        for action in self._parser._actions:
            for opt in action.option_strings:
                for name in names:
                    if opt == name:
                        return True
        return False

    # ── Input flags ──────────────────────────────────────────────────────────
    def test_arg_file(self):
        assert self._has_arg("--file")

    def test_arg_onnx_file(self):
        assert self._has_arg("--onnx_file")

    def test_arg_node_limit(self):
        assert self._has_arg("--node_limit")

    # ── Output flags ─────────────────────────────────────────────────────────
    def test_arg_folder(self):
        assert self._has_arg("--folder")

    def test_arg_code_generation(self):
        assert self._has_arg("--code_generation")

    # ── Device flags ─────────────────────────────────────────────────────────
    def test_arg_device(self):
        assert self._has_arg("--device")

    def test_arg_slr(self):
        assert self._has_arg("--SLR")

    def test_arg_dsp(self):
        assert self._has_arg("--DSP")

    def test_arg_on_chip_mem_size(self):
        assert self._has_arg("--ON_CHIP_MEM_SIZE")

    def test_arg_has_uram(self):
        assert self._has_arg("--has_uram")

    # ── Optimization flags ───────────────────────────────────────────────────
    def test_arg_reuse_nlp(self):
        assert self._has_arg("--reuse_nlp")

    def test_arg_update_shape(self):
        assert self._has_arg("--update_shape")

    def test_arg_not_cyclic_buffer(self):
        assert self._has_arg("--not_cyclic_buffer")

    def test_arg_graph_partitioning(self):
        assert self._has_arg("--graph_partitioning")

    def test_arg_allow_multiple_transfer(self):
        assert self._has_arg("--allow_multiple_transfer")

    # ── HLS execution flags ──────────────────────────────────────────────────
    def test_arg_vitis(self):
        assert self._has_arg("--vitis")

    def test_arg_csim(self):
        assert self._has_arg("--csim")

    def test_arg_print_summary(self):
        assert self._has_arg("--print_summary")

    # ── SLASH backend flags ──────────────────────────────────────────────────
    def test_arg_backend(self):
        assert self._has_arg("--backend")

    def test_arg_export_slash(self):
        assert self._has_arg("--export_slash")

    def test_arg_slash_root(self):
        assert self._has_arg("--slash_root")

    def test_arg_project_name(self):
        assert self._has_arg("--project_name")

    def test_arg_build_slash(self):
        assert self._has_arg("--build_slash")

    def test_arg_dry_run(self):
        assert self._has_arg("--dry_run")


class TestArgparserValidation:
    """Argparser validation and defaults must behave correctly."""

    def setup_method(self):
        self.parser = build_parser()

    def test_backend_default_is_hls(self):
        # Minimal required args
        args = self.parser.parse_args(["--onnx_file", "x.onnx"])
        assert args.backend == "hls"

    def test_device_choices_contains_v80(self):
        for action in self.parser._actions:
            if "--device" in (action.option_strings or []):
                choices = action.choices or []
                assert "Alveo_V80" in choices
                break

    def test_device_choices_contains_u55c(self):
        for action in self.parser._actions:
            if "--device" in (action.option_strings or []):
                choices = action.choices or []
                assert "Alveo_U55C" in choices
                break

    def test_node_limit_help_mentions_matmul(self):
        for action in self.parser._actions:
            if "--node_limit" in (action.option_strings or []):
                help_text = action.help or ""
                assert "MatMul" in help_text or "matmul" in help_text.lower(), (
                    "--node_limit help must mention MatMul (not just 'ONNX nodes')"
                )
                break


class TestValidateArgs:
    """validate_args() must fail fast for invalid combinations."""

    def test_vitis_without_code_generation_raises(self, capsys):
        from main import validate_args
        args = argparse.Namespace(
            file="x.c", onnx_file=None, vitis=True, csim=False,
            code_generation=False, backend="hls", export_slash=False,
            build_slash=False, slash_root=None, dry_run=False,
        )
        with pytest.raises(SystemExit):
            validate_args(args)

    def test_csim_without_code_generation_raises(self, capsys):
        from main import validate_args
        args = argparse.Namespace(
            file="x.c", onnx_file=None, vitis=False, csim=True,
            code_generation=False, backend="hls", export_slash=False,
            build_slash=False, slash_root=None, dry_run=False,
        )
        with pytest.raises(SystemExit):
            validate_args(args)

    def test_backend_slash_without_slash_root_raises(self):
        from main import validate_args
        args = argparse.Namespace(
            file="x.c", onnx_file=None, vitis=False, csim=False,
            code_generation=True, backend="slash", export_slash=False,
            build_slash=False, slash_root=None, dry_run=False,
        )
        with pytest.raises(SystemExit):
            validate_args(args)

    def test_no_file_and_no_onnx_raises(self):
        from main import validate_args
        args = argparse.Namespace(
            file=None, onnx_file=None, vitis=False, csim=False,
            code_generation=False, backend="hls", export_slash=False,
            build_slash=False, slash_root=None, dry_run=False,
        )
        with pytest.raises(SystemExit):
            validate_args(args)
