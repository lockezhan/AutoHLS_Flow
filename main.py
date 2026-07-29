"""
main.py – AutoHLS_Flow: automatic HLS optimization and code generation.

Two-phase flow
--------------
Phase 1 (AutoHLS_Flow):
  ONNX MatMul / Affine C++ → Polyhedral analysis → AMPL/Gurobi NLP →
  Tiling/Unrolling → HLS C++ / Host code / TCL generation → CSIM / C Synthesis

Phase 2 (SLASH/VRT, optional):
  HLS output → SLASH export adapter → CMake build → VRTBIN / Runtime

Device profiles
---------------
Use --device to select a named profile from device_profiles/.
CLI flags --SLR, --DSP, and --ON_CHIP_MEM_SIZE override the profile values
when explicitly set (non-zero).
"""

from __future__ import annotations

import os
import sys
import argparse
import argcomplete
import ast
import pickle

import code_generation
import extract
import analysis as analysis_
import memoryBound
import memoryBoundSplit
import iscc
import subprocess
import parse_vitis_report
import pocc
import code_gen.main as code_gen
import utilities
import splitKernel
from ressources import Ressources, DeviceProfile

AMPL_CMD = "ampl"


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "AutoHLS_Flow: automatic HLS optimization and code generation for FPGA accelerators.\n\n"
            "Use --device to select a device profile from device_profiles/.\n"
            "Explicit resource flags (--SLR, --DSP, --ON_CHIP_MEM_SIZE) override the device profile."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Input ─────────────────────────────────────────────────────────────────
    inp = p.add_argument_group("Input")
    inp.add_argument("--file", type=str, help="Path to the input affine C/C++ kernel")
    inp.add_argument("--onnx_file", type=str, help="Path to the ONNX model file (prototype: MatMul ops only)")
    inp.add_argument("--node_limit", type=int, default=None,
                     help="Limit number of ONNX MatMul nodes to compile (e.g. 4 nodes = 1 Transformer block)")

    # ── Output ────────────────────────────────────────────────────────────────
    out = p.add_argument_group("Output")
    out.add_argument("--folder", type=str, default="hls_output",
                     help="Output folder for generated HLS project (default: hls_output)")
    out.add_argument("--name_function", type=str, default="kernel_nlp",
                     help="Top-level kernel function name (default: kernel_nlp)")
    out.add_argument("--output", type=str, help="Override output file name")
    out.add_argument("--code_generation", action="store_true",
                     help="Enable HLS C++ and host code generation (required for --vitis and --csim)")

    # ── Device & Resources ────────────────────────────────────────────────────
    dev = p.add_argument_group(
        "Device & Resources",
        "Use --device to load a named profile. Explicit flags override the profile.",
    )
    dev.add_argument("--device", type=str, default=None,
                     choices=DeviceProfile.list_available() + [None],
                     help=(
                         f"Target device profile. Known: {', '.join(DeviceProfile.list_available())}. "
                         "Loads part, SLR count, DSP, memory from device_profiles/<name>.json."
                     ))
    dev.add_argument("--SLR", type=int, default=0,
                     help="Number of SLRs. Overrides device profile when non-zero.")
    dev.add_argument("--DSP", type=int, default=0,
                     help="Total DSP slices. Overrides device profile when non-zero.")
    dev.add_argument("--ON_CHIP_MEM_SIZE", type=int, default=0,
                     help="Total on-chip memory in bytes. Overrides device profile when non-zero.")
    dev.add_argument("--MAX_BUFFER_SIZE", type=int, default=0,
                     help="Max on-chip buffer size per array (elements).")
    dev.add_argument("--MAX_UF", type=int, default=0, help="Maximum loop unrolling factor.")
    dev.add_argument("--has_uram", action="store_true",
                     help="Force URAM storage binding for large arrays (auto-enabled for V80 profile).")
    dev.add_argument("--factor", type=float, default=0,
                     help="Resource utilisation factor override (0 = use profile/default).")
    dev.add_argument("--partitioning_max", type=int, default=0)

    # ── Optimization ─────────────────────────────────────────────────────────
    opt = p.add_argument_group("Optimization")
    opt.add_argument("--reuse_nlp", action="store_true",
                     help="Reuse previously computed NLP results (skip AMPL solve).")
    opt.add_argument("--graph_partitioning", action="store_true",
                     help="Enable graph partitioning across SLRs or compute units.")
    opt.add_argument("--no_distribution", action="store_true",
                     help="Disable ISCC-based loop distribution.")
    opt.add_argument("--update_shape", action="store_true",
                     help="Use shape-update variant of HLS code generation.")
    opt.add_argument("--ap_multiple_burst", action="store_true",
                     help="Enable multiple AXI burst access inference.")
    opt.add_argument("--not_cyclic_buffer", action="store_true",
                     help="Disable Ping-Pong/Triple cyclic buffer generation.")
    opt.add_argument("--allow_multiple_transfer", action="store_true",
                     help="Allow multiple DMA transfers per tile iteration.")

    # ── HLS Execution ────────────────────────────────────────────────────────
    hls = p.add_argument_group("HLS Execution")
    hls.add_argument("--vitis", action="store_true",
                     help="Run AMD Vitis HLS C Synthesis after code generation.")
    hls.add_argument("--csim", action="store_true",
                     help="Run AMD Vitis HLS C Simulation after code generation.")
    hls.add_argument("--print_summary", action="store_true",
                     help="Print synthesis resource and timing summary.")

    # ── SLASH/VRT Backend ────────────────────────────────────────────────────
    slash = p.add_argument_group(
        "SLASH/VRT Backend",
        "Options for exporting generated HLS kernels to a SLASH/VRT project for Alveo V80 deployment.",
    )
    slash.add_argument("--backend", type=str, default="hls", choices=["hls", "slash"],
                       help=(
                           "Deployment backend. 'hls': HLS codegen only (default). "
                           "'slash': HLS codegen + SLASH project export."
                       ))
    slash.add_argument("--export_slash", action="store_true",
                       help="Export SLASH project bundle after HLS codegen (same as --backend slash).")
    slash.add_argument("--slash_root", type=str, default=None,
                       help="Path to the SLASH repository root (required for --backend slash / --export_slash).")
    slash.add_argument("--project_name", type=str, default=None,
                       help="Project name for the generated SLASH bundle.")
    slash.add_argument("--build_slash", action="store_true",
                       help=(
                           "Also invoke the SLASH CMake build after exporting. "
                           "WARNING: hardware build can take several hours. "
                           "Requires --slash_root and explicit opt-in."
                       ))
    slash.add_argument("--slash_version", type=str, default="auto",
                       help="SLASH version tag for manifest (default: auto = git SHA)")
    slash.add_argument("--dry_run", action="store_true",
                       help="Simulate all steps without writing files or invoking Vitis HLS.")

    # ── Misc / Advanced ──────────────────────────────────────────────────────
    misc = p.add_argument_group("Misc / Advanced")
    misc.add_argument("--schedule", type=str, nargs="+")
    misc.add_argument("--UB", type=str, nargs="+")
    misc.add_argument("--LB", type=str, nargs="+")
    misc.add_argument("--statements", type=str, nargs="+")
    misc.add_argument("--iterators", type=str, nargs="+")
    misc.add_argument("--headers", type=str, nargs="+")
    misc.add_argument("--arguments", type=str, nargs="+")
    misc.add_argument("--pragmas", type=str, nargs="+")
    misc.add_argument("--pragmas_top", action="store_true")
    misc.add_argument("--not_optimize_burst", action="store_true")

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────
def validate_args(args: argparse.Namespace) -> None:
    """Fail early with clear error messages before any file I/O starts."""
    if not args.file and not args.onnx_file:
        print("[ERROR] Provide --file (C/C++ kernel) or --onnx_file (ONNX model).", file=sys.stderr)
        sys.exit(1)

    if args.vitis and not args.code_generation:
        print("[ERROR] --vitis requires --code_generation.", file=sys.stderr)
        sys.exit(1)

    if args.csim and not args.code_generation:
        print("[ERROR] --csim requires --code_generation.", file=sys.stderr)
        sys.exit(1)

    need_slash = args.backend == "slash" or args.export_slash or args.build_slash
    if need_slash and not args.slash_root:
        if not args.dry_run:
            print("[ERROR] --backend slash / --export_slash / --build_slash require --slash_root.", file=sys.stderr)
            sys.exit(1)

    if args.build_slash and not need_slash:
        print("[ERROR] --build_slash requires --slash_root.", file=sys.stderr)
        sys.exit(1)

    if need_slash and not args.code_generation:
        print("[ERROR] SLASH Backend requires --code_generation.", file=sys.stderr)
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Resource initialisation
# ─────────────────────────────────────────────────────────────────────────────
def init_resources(args: argparse.Namespace) -> tuple[Ressources, dict | None]:
    """Load device profile (if any) and apply CLI overrides."""
    profile: dict | None = None
    if args.device is not None:
        profile = DeviceProfile.load(args.device)
        print(
            f"[Device Profile] Loaded '{args.device}': "
            f"part={profile['hls_part']}, SLR={profile['slr_count']}, "
            f"DSP={profile['dsp']}, mem={profile['on_chip_mem_bytes']} bytes, "
            f"has_uram={profile['has_uram']}, backend={profile['deployment_backend']}"
        )

    res = Ressources(profile=profile)

    # CLI overrides – always win over profile values
    if args.SLR != 0:
        res.SLR = args.SLR
    if args.DSP != 0:
        res.DSP = args.DSP
    if args.ON_CHIP_MEM_SIZE != 0:
        res.ON_CHIP_MEM_SIZE = args.ON_CHIP_MEM_SIZE
    if args.MAX_BUFFER_SIZE != 0:
        res.MAX_BUFFER_SIZE = args.MAX_BUFFER_SIZE
    if args.MAX_UF != 0:
        res.MAX_UF = args.MAX_UF
    if args.factor != 0:
        res.factor = args.factor
    if args.partitioning_max != 0:
        res.partitioning_max = args.partitioning_max
    if args.has_uram:
        res.has_uram = True

    # Derived per-SLR values (must be recomputed after overrides)
    res.DSP_per_SLR = int(res.DSP / res.SLR)
    res.BRAM_per_SLR = int(res.BRAM / res.SLR)
    res.MEM_PER_SLR = int(res.ON_CHIP_MEM_SIZE / res.SLR)

    return res, profile


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = build_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    # ── Early validation ──────────────────────────────────────────────────────
    validate_args(args)

    # ── Resolve output paths ──────────────────────────────────────────────────
    output = f"{args.folder}/src/output.cpp"
    host_name = f"{args.folder}/src/host.cpp"

    # ── Create project directories ────────────────────────────────────────────
    if args.dry_run:
        print(f"[DRY RUN] Would create project in: {args.folder}/")
    else:
        os.makedirs(args.folder, exist_ok=True)
        os.makedirs(f"{args.folder}/tmp", exist_ok=True)
        os.makedirs(f"{args.folder}/src", exist_ok=True)
        os.makedirs(f"{args.folder}/tcl_scripts", exist_ok=True)

        os.system(f"cp script/constrain_blocks.tcl {args.folder}/tcl_scripts")
        os.system(f"cp script/phys_opt_loop.tcl {args.folder}/tcl_scripts")
        os.system(f"cp script/print_CR.tcl {args.folder}/tcl_scripts")
        os.system(f"cp script/hls_pre.tcl {args.folder}/tcl_scripts")
        os.system(f"cp script/post_place_qor.tcl {args.folder}/tcl_scripts")
        os.system(f"cp script/print_CR_utilization.tcl {args.folder}/tcl_scripts")
        os.system(f"cp script/xcl2* {args.folder}/src")

        # Device-aware script selection
        if args.device == "Alveo_V80":
            os.system(f"cp script/hls_run_v80.sh {args.folder}/hls_run.sh")
        else:
            os.system(f"cp script/hls_run.sh {args.folder}/")

        os.system(f"cp script/Makefile {args.folder}/")
        os.system(f"cp script/build.tcl {args.folder}/")
        os.system(f"cp script/hls_config_slr0.cfg {args.folder}/")

    nlp_file = f"{args.folder}/nlp.mod"
    nlp_log = f"{args.folder}/nlp.log"

    # ── Resource initialisation ───────────────────────────────────────────────
    res, profile = init_resources(args)

    # ── Frontend parsing ──────────────────────────────────────────────────────
    if args.onnx_file:
        import onnx_frontend
        nodes = onnx_frontend.parse_onnx_to_hls(args.onnx_file, node_limit=args.node_limit)

        schedule = []
        dic = {}
        operations = []
        operation_list = []
        arrays_size = {}
        dep = []

        for id_statement, node in enumerate(nodes):
            iterators_list = [b[0] for b in node.loop_bounds]
            sched_entry = [id_statement]
            for it in iterators_list:
                sched_entry.append(it)
                sched_entry.append(0)
            pragma = ["" for _ in iterators_list]
            schedule.append([f"S{id_statement}", sched_entry, pragma])

            op_dict = {"+": 0, "-": 0, "*": 0, "/": 0}
            op_list = []
            for op in op_dict:
                count = node.computation.count(op)
                op_dict[op] += count
                op_list.extend([op] * count)
            operations.append(op_dict)
            operation_list.append(op_list)

            for inp, shape in node.inputs:
                arrays_size[inp] = shape
            for out, shape in node.outputs:
                arrays_size[out] = shape

            LB_dict, UB_dict, TC_dict = {}, {}, {}
            for b in node.loop_bounds:
                LB_dict[b[0]] = b[1]
                UB_dict[b[0]] = b[2] - 1
                TC_dict[b[0]] = b[2] - b[1]

            dic[id_statement] = {
                "read": node.read_access,
                "write": node.write_access,
                "statement_body": node.computation,
                "TC": TC_dict,
                "LB": LB_dict,
                "UB": UB_dict,
                "LB_": LB_dict,
                "UB_": UB_dict,
                "constraint": [],
            }
        iscc_ = None
    else:
        # Affine C/C++ kernel path
        if not args.no_distribution:
            schedule, dic, operations, arrays_size, dep, operation_list = extract.compute_statement(
                args.folder, args.file
            )
            iscc_ = iscc.ISCC(
                args.no_distribution, args.folder, schedule, dic,
                operations, arrays_size, dep, operation_list,
            )
        else:
            os.system(f"cp {args.file} {args.folder}/new.cpp")
            with open(f"{args.folder}/new.cpp", "r") as fh:
                lines = fh.readlines()
            for i, line in enumerate(lines):
                if "void" in line and "(" in line and ")" in line:
                    lines[i] = "void kernel_nlp(" + line.split("(")[1]
                    break
            with open(f"{args.folder}/new.cpp", "w") as fh:
                fh.writelines(lines)

        schedule, dic, operations, arrays_size, dep, operation_list = extract.compute_statement(
            args.folder, f"{args.folder}/new.cpp"
        )

    # ── Analysis ──────────────────────────────────────────────────────────────
    analysis = analysis_.Analysis(schedule, dic, operations, arrays_size, dep, operation_list)
    UB = analysis.UB
    LB = analysis.LB
    statements = analysis.statements
    iterators = analysis.iterators
    schedule = analysis.only_schedule

    headers = ["ap_int.h", "hls_stream.h", "hls_vector.h", "cstring"]
    arguments = []

    if not args.onnx_file:
        with open(args.file, "r") as fh:
            lines = fh.readlines()
        for line in lines:
            if "void" in line and "(" in line and ")" in line:
                cte = line.split("(")[1].split(")")[0].split(",")
                for cc in cte:
                    if "[" not in cc:
                        arguments.append(cc)
                break

    arguments += analysis.arguments
    name_function = args.name_function or "kernel_nlp"
    pragmas = [[] for _ in range(len(UB))]
    pragmas_top = False

    # ── NLP Optimisation ──────────────────────────────────────────────────────
    if not args.reuse_nlp:
        if args.graph_partitioning:
            split = splitKernel.Identify(
                "new.cpp", nlp_file, analysis, schedule, UB, LB,
                statements, iterators, output, headers, arguments,
                name_function, pragmas, pragmas_top,
            )
            memoryBoundSplit.memoryBound(
                res, args.folder, split, "new.cpp", nlp_file, analysis,
                schedule, UB, LB, statements, iterators, output,
                headers, arguments, name_function, pragmas, pragmas_top,
            )
            utilities.run_ampl_py(args.folder, "nlp.mod", "nlp.log")
            splitKernel.splitKernel(
                "new.cpp", nlp_file, analysis, schedule, UB, LB,
                statements, iterators, output, headers, arguments,
                name_function, pragmas, pragmas_top,
            )
        else:
            memoryBound.memoryBound(
                res, args.folder, args.allow_multiple_transfer,
                args.ap_multiple_burst, "new.cpp", nlp_file, analysis,
                schedule, UB, LB, statements, iterators, output,
                headers, arguments, name_function, pragmas, pragmas_top,
            )
            utilities.run_ampl_py(args.folder, "nlp.mod", "nlp.log")

    # ── HLS Code Generation ───────────────────────────────────────────────────
    if args.code_generation:
        if args.dry_run:
            print(f"[DRY RUN] Would generate HLS code in: {output}")
        else:
            code_gen.code_gen(
                args.update_shape, res.SLR, nlp_file, nlp_log,
                args.file, output, host_name, schedule, analysis,
                getattr(res, "has_uram", False), args.not_cyclic_buffer
            )

            # Generate device-aware TCL scripts
            from code_gen.write_tcl import generate_tcl_from_profile, generate_tcl, generate_csim
            tcl_vitis = f"{args.folder}/src/vitis.tcl"
            tcl_csim  = f"{args.folder}/src/csim.tcl"
            if profile is not None:
                generate_tcl_from_profile(tcl_vitis, profile, mode="csynth")
                generate_tcl_from_profile(tcl_csim, profile, mode="csim")
            else:
                os.system(f"cp script/vitis.tcl {tcl_vitis}")
                os.system(f"cp script/csim.tcl  {tcl_csim}")

            # Format generated code
            for fmt_path in [output, host_name]:
                try:
                    os.system(f"clang-format -i {fmt_path}")
                except Exception:
                    pass
            for id_slr in range(res.SLR):
                slr_path = output.replace("output.cpp", f"slr{id_slr}.cpp")
                try:
                    os.system(f"clang-format -i {slr_path}")
                except Exception:
                    pass
            for extra in [output.split(".")[0] + ".h", host_name.replace("host.cpp", "csim.cpp")]:
                try:
                    os.system(f"clang-format -i {extra}")
                except Exception:
                    pass

            print(f"[AutoHLS_Flow] Files generated in {args.folder}/")
    else:
        print("[AutoHLS_Flow] --code_generation not specified: skipping HLS C++ output.")

    # ── Vitis HLS Execution ───────────────────────────────────────────────────
    if args.csim:
        if args.dry_run:
            print("[DRY RUN] Would run Vitis HLS csim.")
        else:
            utilities.run_vitis_hls("csim.tcl", f"{args.folder}/src")

    if args.vitis:
        if args.dry_run:
            print("[DRY RUN] Would run Vitis HLS csynth.")
        else:
            utilities.run_vitis_hls("vitis.tcl", f"{args.folder}/src")
            if args.print_summary:
                utilities.print_summary(args.folder, args.file)

    # ── SLASH Export ──────────────────────────────────────────────────────────
    do_slash_export = args.backend == "slash" or args.export_slash or args.build_slash
    if do_slash_export:
        from integrations.slash.export_to_slash import SlashExporter
        slash_out = f"slash_projects/{args.project_name or 'autohls_project'}"
        exporter = SlashExporter(
            hls_project=args.folder,
            slash_root=args.slash_root,
            project_name=args.project_name or "autohls_project",
            output_dir=slash_out,
            device_profile=profile,
            target_frequency=profile["default_freq_mhz"] if profile else 300,
            slash_version=args.slash_version,
            dry_run=args.dry_run,
            verbose=True,
        )
        if args.dry_run:
            exporter.dry_run_export()
        else:
            exporter.export()

        if args.build_slash and not args.dry_run:
            print("\n[SLASH] Invoking CMake build – this may take several hours.")
            print("[SLASH] Full log: see logs/ in the SLASH project directory.")
            ret = os.system(f"cd {slash_out} && cmake -B build -S . -G Ninja -DSLASH_USE_REPO=ON -DSLASH_REPO_ROOT={args.slash_root} && cmake --build build")
            if ret != 0:
                print("[ERROR] SLASH CMake build failed. Check logs above.", file=sys.stderr)
                sys.exit(ret)