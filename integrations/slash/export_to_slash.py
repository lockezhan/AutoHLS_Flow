"""
integrations/slash/export_to_slash.py
======================================
AutoHLS_Flow → SLASH/VRT project export adapter.

This tool bridges the gap between the AutoHLS_Flow HLS code generation phase
and the SLASH/VRT physical deployment phase on AMD Alveo V80.

Responsibility boundary
-----------------------
AutoHLS_Flow (this tool):   Frontend parsing, NLP optimization, HLS C++ generation, CSIM/csynth
SLASH / VRT (external):     Segmented Configuration, VRTBIN build, runtime deployment, board execution

The SLASH/VRT backend is NOT part of AutoHLS_Flow.  This adapter generates
the necessary SLASH project files from AutoHLS_Flow output, but does NOT
modify the SLASH repository itself.

Usage
-----
    python integrations/slash/export_to_slash.py \\
      --hls-project hls_output_demo \\
      --slash-root /path/to/SLASH \\
      --project-name autohls_attention \\
      --output-dir slash_projects/autohls_attention \\
      [--dry-run] [--force] [--verbose] \\
      [--target-frequency 300] [--slash-version auto]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

# ── Verilog/VHDL reserved words that must not appear as kernel names ──────────
_RESERVED_WORDS = frozenset({
    "module", "endmodule", "input", "output", "wire", "reg", "always",
    "begin", "end", "if", "else", "case", "default", "assign", "initial",
    "integer", "parameter", "localparam", "logic", "bit", "byte",
    "function", "endfunction", "task", "endtask",
    # VHDL
    "entity", "architecture", "port", "signal", "process", "component",
    "generic", "variable", "constant", "library", "use", "std_logic",
})

# ── Minimum files expected in a valid AutoHLS_Flow project ────────────────────
_REQUIRED_FILES = ["src/output.h"]
_REQUIRED_SOURCES = ["src/output.cpp"]  # OR at least one slrX.cpp


class ExportError(RuntimeError):
    """Raised when a pre-flight check fails."""


class SlashExporter:
    """
    Export AutoHLS_Flow-generated HLS output to a SLASH/VRT project bundle.

    Parameters
    ----------
    hls_project : str | Path
        Path to the AutoHLS_Flow output folder (e.g. ``hls_output_demo``).
    slash_root : str | Path | None
        Path to the SLASH repository root.  Required unless ``dry_run`` is True.
    project_name : str
        Name for the generated SLASH project.
    output_dir : str | Path
        Destination directory for the SLASH project bundle.
    device_profile : dict | None
        Device profile dict from ``DeviceProfile.load()``.  Provides ``hls_part``
        and ``default_freq_mhz``.  Falls back to V80 defaults when None.
    target_frequency : int
        HLS target frequency in MHz (overrides device profile if given explicitly).
    dry_run : bool
        If True, perform all validation checks but write no files.
    verbose : bool
        Print detailed progress messages.
    force : bool
        Overwrite existing output directory without prompting.
    """

    _DEFAULT_PART = "xcv80-lsva4737-2MHP-e-S"
    _DEFAULT_FREQ = 300

    def __init__(
        self,
        hls_project: str | Path,
        project_name: str,
        output_dir: str | Path,
        slash_root: Optional[str | Path] = None,
        device_profile: Optional[dict] = None,
        target_frequency: Optional[int] = None,
        dry_run: bool = False,
        verbose: bool = False,
        force: bool = False,
    ) -> None:
        self.hls_project = Path(hls_project).resolve()
        self.slash_root = Path(slash_root).resolve() if slash_root else None
        self.project_name = project_name
        self.output_dir = Path(output_dir).resolve()
        self.device_profile = device_profile or {}
        self.dry_run = dry_run
        self.verbose = verbose
        self.force = force

        self.part = self.device_profile.get("hls_part", self._DEFAULT_PART)
        self.freq = target_frequency or self.device_profile.get("default_freq_mhz", self._DEFAULT_FREQ)

    # ── Public API ────────────────────────────────────────────────────────────

    def export(self) -> Path:
        """Run all checks and generate the SLASH project bundle."""
        self._log(f"[SLASH Export] Starting export for project '{self.project_name}'")
        self._preflight_checks()
        hls_sources = self._discover_hls_sources()
        kernel_name = self._extract_kernel_name(hls_sources["top"])
        self._check_reserved_word(kernel_name)
        k2k_path = self._find_k2k()
        nlp_tiling = self._parse_nlp_results()
        manifest = self._build_manifest(kernel_name, hls_sources, k2k_path, nlp_tiling)

        if self.dry_run:
            self._log("[DRY RUN] Validation passed. No files written.")
            self._print_next_steps()
            return self.output_dir

        self._prepare_output_dir()
        generated = self._write_bundle(manifest, hls_sources, k2k_path)
        self._write_manifest_json(manifest)
        self._log(f"[SLASH Export] Success – {len(generated)} files written to: {self.output_dir}")
        self._print_next_steps()
        return self.output_dir

    def dry_run_export(self) -> None:
        """Alias for export() with dry_run forced True."""
        orig = self.dry_run
        self.dry_run = True
        try:
            self.export()
        finally:
            self.dry_run = orig

    # ── Pre-flight checks ─────────────────────────────────────────────────────

    def _preflight_checks(self) -> None:
        """Raise ExportError with a specific message for every problem found."""
        errors: list[str] = []

        # 1. HLS project directory
        if not self.hls_project.exists():
            raise ExportError(
                f"HLS project directory not found: {self.hls_project}\n"
                "  Run: python main.py --code_generation --folder <folder>"
            )

        # 2. Required header
        for rel in _REQUIRED_FILES:
            if not (self.hls_project / rel).exists():
                errors.append(f"Missing required file: {self.hls_project / rel}")

        # 3. At least one HLS source
        has_output = (self.hls_project / "src/output.cpp").exists()
        has_slr = any((self.hls_project / "src").glob("slr*.cpp"))
        if not (has_output or has_slr):
            errors.append(
                f"No HLS kernel source found in {self.hls_project}/src/. "
                "Expected output.cpp or slr0.cpp / slr1.cpp / ..."
            )

        # 4. NLP results
        if not (self.hls_project / "nlp.log").exists():
            errors.append(
                f"NLP results missing: {self.hls_project}/nlp.log\n"
                "  Ensure the optimization step completed successfully."
            )

        # 5. SLASH root (not required for dry-run)
        if self.slash_root is not None:
            if not self.slash_root.exists():
                errors.append(
                    f"SLASH root directory not found: {self.slash_root}\n"
                    "  Clone SLASH from: https://github.com/hpc-aulmamei/SLASH.git"
                )
            else:
                abs_shell = self.slash_root / "linker/resources/abstract_shell/abs_shell_slash.dcp"
                if not abs_shell.exists():
                    errors.append(
                        f"SLASH abstract shell DCP missing: {abs_shell}\n"
                        "  Run the one-time installer:\n"
                        f"  cd {self.slash_root}/linker/src && "
                        "python3 main.py install --build-dir "
                        f"{self.slash_root}/linker/resources/abstract_shell_build"
                    )

        if errors:
            msg = "\n".join(f"  - {e}" for e in errors)
            raise ExportError(f"Pre-flight checks failed:\n{msg}")

        self._log("[SLASH Export] Pre-flight checks passed.")

    # ── Source discovery ──────────────────────────────────────────────────────

    def _discover_hls_sources(self) -> dict:
        """Return a dict with 'top', 'slrs', 'header' paths."""
        src = self.hls_project / "src"
        top = src / "output.cpp"
        slrs = sorted(src.glob("slr*.cpp"))
        header = src / "output.h"
        return {"top": top, "slrs": slrs, "header": header}

    def _extract_kernel_name(self, source_path: Path) -> str:
        """Parse the top-level kernel function name from the HLS source."""
        if not source_path.exists():
            return "kernel_nlp"
        text = source_path.read_text(encoding="utf-8")
        # Look for 'void kernel_name(' pattern
        m = re.search(r"void\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", text)
        return m.group(1) if m else "kernel_nlp"

    def _check_reserved_word(self, kernel_name: str) -> None:
        if kernel_name.lower() in _RESERVED_WORDS:
            raise ExportError(
                f"Kernel function name '{kernel_name}' is a Verilog/VHDL reserved word. "
                "Rename the kernel function before exporting."
            )

    def _find_k2k(self) -> Optional[Path]:
        k2k = self.hls_project / "src/k2k.cfg"
        if k2k.exists():
            self._log(f"[SLASH Export] Found k2k.cfg: {k2k}")
            return k2k
        self._log("[SLASH Export] No k2k.cfg found – stream connectivity will be empty.")
        return None

    def _parse_nlp_results(self) -> dict:
        """Extract tiling factors from nlp.log."""
        nlp_log = self.hls_project / "nlp.log"
        tiling: dict[str, int] = {}
        if not nlp_log.exists():
            return tiling
        for line in nlp_log.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if "=" in line and line.startswith("TC"):
                k, _, v = line.partition("=")
                try:
                    tiling[k.strip()] = int(v.strip())
                except ValueError:
                    pass
        return tiling

    # ── Manifest construction ─────────────────────────────────────────────────

    def _get_git_sha(self, repo: Path) -> str:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(repo), capture_output=True, text=True, check=True,
            )
            return result.stdout.strip()
        except Exception:
            return "unknown"

    def _build_manifest(
        self,
        kernel_name: str,
        hls_sources: dict,
        k2k_path: Optional[Path],
        nlp_tiling: dict,
    ) -> dict:
        slr_kernels = [f"{kernel_name}_slr{i}" for i in range(len(hls_sources["slrs"]))]
        if not slr_kernels:
            slr_kernels = [kernel_name]

        slr_map = {
            f"slr{i}": f"{slr_kernels[i]}_0"
            for i in range(len(slr_kernels))
        }

        autohls_sha = self._get_git_sha(Path(__file__).parents[2])
        slash_sha = (
            self._get_git_sha(self.slash_root) if self.slash_root else "not-available"
        )

        return {
            "schema_version": "1.0",
            "generation_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "autohls_flow_version": autohls_sha,
            "slash_bridge_version": slash_sha,
            "project_name": self.project_name,
            "kernel_name": kernel_name,
            "kernel_count": len(slr_kernels),
            "kernel_functions": slr_kernels,
            "interfaces": {
                "axi_lite": "s_axi_control",
                "axi_mm": "m_axi (auto-discovered from kernel signature)",
                "axi_stream": "hls::stream (auto-inferred from FIFOs)",
            },
            "target_freq_mhz": self.freq,
            "hls_part": self.part,
            "data_type": "float32",
            "slr_mapping": slr_map,
            "stream_connectivity": str(k2k_path) if k2k_path else "none",
            "nlp_tiling": nlp_tiling,
            "deployment_backend": self.device_profile.get("deployment_backend", "SLASH/VRT"),
            "synthesis_status": "C Synthesis Estimate (Vitis HLS). Board-level execution not yet verified.",
        }

    # ── Bundle writing ────────────────────────────────────────────────────────

    def _prepare_output_dir(self) -> None:
        if self.output_dir.exists():
            if self.force:
                shutil.rmtree(self.output_dir)
            else:
                raise ExportError(
                    f"Output directory already exists: {self.output_dir}\n"
                    "  Use --force to overwrite."
                )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _write_bundle(
        self, manifest: dict, hls_sources: dict, k2k_path: Optional[Path]
    ) -> list[Path]:
        generated: list[Path] = []

        # Try to use prometheus_slash_bridge if available
        try:
            slash_bridge_root = Path("/home/Ryan/prometheus_slash_bridge")
            if slash_bridge_root.exists():
                sys.path.insert(0, str(slash_bridge_root))
            from prometheus_slash_bridge.bridge import SlashBridge
            from prometheus_slash_bridge.models import BridgeManifest

            bridge_manifest = BridgeManifest(
                project_name=manifest["project_name"],
                kernel_name=manifest["kernel_name"],
                top_source=str(hls_sources["top"]),
                part=self.part,
                clock=f"{round(1000/self.freq, 2)}ns",
                slash_repo=str(self.slash_root) if self.slash_root else None,
                device=self.part,
                clock_kernel=f"{manifest['kernel_name']}_slr0_0",
                clock_freq_hz=self.freq * 1_000_000,
                hls_output="output.cpp",
                output_name=self.project_name,
            )
            bridge = SlashBridge(bridge_manifest)
            if k2k_path:
                from prometheus_slash_bridge.parser import parse_k2k_file
                bridge.connectivity = parse_k2k_file(str(k2k_path))

            generated_paths = bridge.write_bundle(self.output_dir)
            generated.extend(generated_paths)
            self._log("[SLASH Export] Used prometheus_slash_bridge for bundle generation.")
        except ImportError:
            # Fallback: write minimal bundle without the bridge
            self._log("[SLASH Export] prometheus_slash_bridge not available – using built-in templates.")
            generated.extend(self._write_fallback_bundle(manifest, hls_sources, k2k_path))

        # Always copy HLS source files
        for slr_path in hls_sources["slrs"]:
            dest = self.output_dir / slr_path.name
            shutil.copy2(slr_path, dest)
            generated.append(dest)

        if hls_sources["header"].exists():
            dest = self.output_dir / hls_sources["header"].name
            shutil.copy2(hls_sources["header"], dest)
            generated.append(dest)

        # Copy output_2.h if present
        output2 = hls_sources["top"].parent / "output_2.h"
        if output2.exists():
            dest = self.output_dir / "output_2.h"
            shutil.copy2(output2, dest)
            generated.append(dest)

        return generated

    def _write_fallback_bundle(
        self, manifest: dict, hls_sources: dict, k2k_path: Optional[Path]
    ) -> list[Path]:
        """Write a minimal SLASH bundle without prometheus_slash_bridge."""
        generated: list[Path] = []

        # output.cfg
        output_cfg = self.output_dir / "output.cfg"
        output_cfg.write_text(
            f"part={self.part}\n\n"
            "[hls]\n"
            "flow_target=vivado\n"
            f"syn.top={manifest['kernel_name']}\n"
            "syn.file=output.cpp\n"
            f"clock={round(1000/self.freq, 2)}ns\n\n"
            "package.output.format=ip_catalog\n"
            "package.output.syn=false\n",
            encoding="utf-8",
        )
        generated.append(output_cfg)

        # config.cfg  (connectivity)
        lines = ["[connectivity]\n"]
        for i, kfn in enumerate(manifest["kernel_functions"]):
            lines.append(f"nk={kfn}:1:{kfn}_0\n")
        for i in range(len(manifest["kernel_functions"])):
            lines.append(f"slr={manifest['kernel_functions'][i]}_0:SLR{i}\n")
        config_cfg = self.output_dir / "config.cfg"
        config_cfg.write_text("".join(lines), encoding="utf-8")
        generated.append(config_cfg)

        # slr_constraints.tcl  (minimal)
        slr_tcl = self.output_dir / "slr_constraints.tcl"
        slr_tcl.write_text(
            "# Auto-generated SLR constraints by AutoHLS_Flow SLASH export adapter\n",
            encoding="utf-8",
        )
        generated.append(slr_tcl)

        # CMakeLists.txt  (minimal)
        cmake = self.output_dir / "CMakeLists.txt"
        cmake.write_text(
            "cmake_minimum_required(VERSION 3.20)\n"
            f"project({manifest['project_name']} LANGUAGES CXX)\n\n"
            "set(CMAKE_CXX_STANDARD 20)\n\n"
            "option(SLASH_USE_REPO \"Build against local repo tree\" OFF)\n\n"
            "if(SLASH_USE_REPO)\n"
            "  set(SLASH_REPO_ROOT \"${SLASH_REPO_ROOT}\" CACHE PATH \"\")\n"
            "  list(APPEND CMAKE_MODULE_PATH \"${SLASH_REPO_ROOT}/cmake\")\n"
            "  include(SlashTools)\n"
            "  set(VRT_INCLUDE_VRTD ON)\n"
            "  set(VRTD_INCLUDE_LIBSLASH ON)\n"
            "  add_subdirectory(${SLASH_REPO_ROOT}/vrt ${CMAKE_CURRENT_BINARY_DIR}/vrt)\n"
            "  set(_VRT_LIBS vrt)\n"
            "else()\n"
            "  find_package(vrt REQUIRED CONFIG)\n"
            "  set(_VRT_LIBS vrt::vrt)\n"
            "endif()\n\n"
            f"build_hls(TARGET hls_{manifest['kernel_name']}\n"
            "    CPP \"${CMAKE_CURRENT_SOURCE_DIR}/output.cpp\"\n"
            "    CFG \"${CMAKE_CURRENT_SOURCE_DIR}/output.cfg\"\n"
            f"    DEVICE \"{self.part}\")\n\n"
            f"add_vbin(TARGET \"{manifest['project_name']}_hw\" PLATFORM \"hw\"\n"
            "    CFG \"${CMAKE_CURRENT_SOURCE_DIR}/config.cfg\"\n"
            f"    KERNELS ${{hls_{manifest['kernel_name']}_COMPONENT_XML}})\n\n"
            "add_executable(host host.cpp)\n"
            "target_link_libraries(host PRIVATE ${_VRT_LIBS})\n",
            encoding="utf-8",
        )
        generated.append(cmake)

        # host.cpp  (minimal VRT skeleton)
        host = self.output_dir / "host.cpp"
        host.write_text(
            '#include <api/buffer.hpp>\n'
            '#include <api/device.hpp>\n'
            '#include <api/kernel.hpp>\n'
            '#include <iostream>\n'
            '#include <string>\n\n'
            'int main(int argc, char* argv[]) {\n'
            '    if (argc < 3) {\n'
            f'        std::cerr << "Usage: " << argv[0] << " <BDF> <{manifest["project_name"]}.vrtbin>" << std::endl;\n'
            '        return 1;\n'
            '    }\n'
            '    vrt::Device device(argv[1], argv[2]);\n'
            f'    vrt::Kernel kernel(device, "{manifest["kernel_name"]}_slr0_0");\n'
            '    kernel.start();\n'
            '    kernel.wait();\n'
            '    device.cleanup();\n'
            '    return 0;\n'
            '}\n',
            encoding="utf-8",
        )
        generated.append(host)

        # run_v80.sh
        run_sh = self.output_dir / "run_v80.sh"
        run_sh.write_text(
            '#!/usr/bin/env bash\n'
            'set -euo pipefail\n\n'
            'if [[ $# -lt 2 ]]; then\n'
            '  echo "Usage: $0 <BDF> <SLASH_REPO_ROOT>"\n'
            '  exit 1\n'
            'fi\n\n'
            'BDF="$1"\n'
            'SLASH_REPO_ROOT="$2"\n'
            'LOG_DIR="${PWD}/logs"\n'
            'mkdir -p "${LOG_DIR}"\n'
            'LOG_FILE="${LOG_DIR}/build_$(date +%Y%m%d_%H%M%S).log"\n\n'
            'cmake -B build -S . -G Ninja \\\n'
            '  -DSLASH_USE_REPO=ON \\\n'
            '  -DSLASH_REPO_ROOT="${SLASH_REPO_ROOT}" \\\n'
            '  -DVRT_INCLUDE_VRTD=ON \\\n'
            '  -DVRTD_INCLUDE_LIBSLASH=ON 2>&1 | tee -a "${LOG_FILE}"\n\n'
            f'cmake --build build --target hls_{manifest["kernel_name"]} 2>&1 | tee -a "${{LOG_FILE}}"\n'
            f'cmake --build build --target {manifest["project_name"]}_hw 2>&1 | tee -a "${{LOG_FILE}}"\n\n'
            f'v80-smi program build/{manifest["project_name"]}_hw.vbin -d ${{BDF}} 2>&1 | tee -a "${{LOG_FILE}}"\n'
            f'./build/host ${{BDF}} build/{manifest["project_name"]}_hw.vbin 2>&1 | tee -a "${{LOG_FILE}}"\n',
            encoding="utf-8",
        )
        run_sh.chmod(0o755)
        generated.append(run_sh)

        # Copy HLS source
        if hls_sources["top"].exists():
            dest = self.output_dir / "output.cpp"
            shutil.copy2(hls_sources["top"], dest)
            generated.append(dest)

        return generated

    def _write_manifest_json(self, manifest: dict) -> None:
        dest = self.output_dir / "autohls_flow_manifest.json"
        dest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        self._log(f"[SLASH Export] Manifest written: {dest}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _print_next_steps(self) -> None:
        print("\n" + "=" * 70)
        print(" AutoHLS_Flow → SLASH Export: Next Steps")
        print("=" * 70)
        if self.dry_run:
            print(" [DRY RUN] All checks passed. Run without --dry-run to write files.")
        else:
            out = self.output_dir
            slash = self.slash_root or "<SLASH_REPO_ROOT>"
            print(f" 1. SLASH project written to: {out}")
            print(f" 2. Build and deploy:")
            print(f"      cd {out}")
            print(f"      bash run_v80.sh <BDF> {slash}")
            print(" 3. Or step-by-step CMake:")
            print(f"      cmake -B build -S . -G Ninja -DSLASH_USE_REPO=ON -DSLASH_REPO_ROOT={slash}")
            print(f"      cmake --build build --target hls_{'{kernel_name}'}")
            print(f"      cmake --build build --target <project>_hw")
            print(f"      v80-smi program build/<project>_hw.vbin -d <BDF>")
        print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(
        description="Export AutoHLS_Flow HLS output to a SLASH/VRT project bundle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--hls-project", required=True, metavar="DIR",
                   help="AutoHLS_Flow output directory (e.g. hls_output_demo)")
    p.add_argument("--slash-root", default=None, metavar="DIR",
                   help="SLASH repository root (required unless --dry-run)")
    p.add_argument("--project-name", default="autohls_project", metavar="NAME",
                   help="Project name for the SLASH bundle (default: autohls_project)")
    p.add_argument("--output-dir", required=True, metavar="DIR",
                   help="Destination directory for the generated SLASH project")
    p.add_argument("--target-frequency", type=int, default=None, metavar="MHZ",
                   help="HLS target frequency in MHz (default: from device profile or 300)")
    p.add_argument("--slash-version", default="auto",
                   help="SLASH version tag for manifest (default: auto = git SHA)")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate inputs and show what would be generated without writing files")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing output directory")
    p.add_argument("--verbose", action="store_true", help="Print detailed progress")
    args = p.parse_args()

    if not args.dry_run and not args.slash_root:
        p.error("--slash-root is required unless --dry-run is specified.")

    exporter = SlashExporter(
        hls_project=args.hls_project,
        slash_root=args.slash_root,
        project_name=args.project_name,
        output_dir=args.output_dir,
        target_frequency=args.target_frequency,
        dry_run=args.dry_run,
        verbose=args.verbose,
        force=args.force,
    )

    try:
        exporter.export()
    except ExportError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
