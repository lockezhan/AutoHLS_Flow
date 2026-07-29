"""
code_gen/write_tcl.py – Device-aware TCL script generation for AutoHLS_Flow.

All generated TCL files use the device part and frequency from the device profile
loaded at runtime.  The string ``xcu55c-fsvh2892-2L-e`` must NEVER appear in
output directed at Alveo V80 or any other target.
"""

from __future__ import annotations

# ── Fallback defaults (used only when no device profile is provided) ──────────
_DEFAULT_PART_V80  = "xcv80-lsva4737-2MHP-e-S"
_DEFAULT_PART_U55C = "xcu55c-fsvh2892-2L-e"  # U55C only – never used for V80


def generate_tcl(
    output_file: str,
    part: str = _DEFAULT_PART_V80,
    freq: str = "300",
) -> None:
    """Generate a Vitis HLS csynth TCL script.

    Parameters
    ----------
    output_file : str
        Destination path for the TCL file.
    part : str
        FPGA part string from the device profile (e.g. ``xcv80-lsva4737-2MHP-e-S``).
        Defaults to the V80 part so that omitting this argument is safe for V80 projects.
    freq : str
        Target frequency in MHz (e.g. ``"300"``).
    """
    content = f"""\
catch {{::common::set_param -quiet hls.xocc.mode csynth}};

open_project kernel_nlp
set_top kernel_nlp

add_files "output.cpp"
open_solution -flow_target vivado solution
set_part {part}

create_clock -period {freq}MHz -name default

config_dataflow -strict_mode warning
config_export -disable_deadlock_detection=true

config_rtl -m_axi_conservative_mode=1
config_interface -m_axi_addr64
config_interface -m_axi_auto_max_ports=0
config_export -format ip_catalog -ipname kernel_nlp
config_compile -unsafe_math_optimizations

csynth_design
close_project
puts "HLS csynth completed successfully"
exit
"""
    with open(output_file, "w", encoding="utf-8") as fh:
        fh.write(content)


def generate_csim(
    output_file: str,
    part: str = _DEFAULT_PART_V80,
    freq: str = "300",
) -> None:
    """Generate a Vitis HLS csim TCL script.

    Parameters
    ----------
    output_file : str
        Destination path for the TCL file.
    part : str
        FPGA part string from the device profile.
    freq : str
        Target frequency in MHz.
    """
    content = f"""\
catch {{::common::set_param -quiet hls.xocc.mode csynth}};

open_project csim.prj
set_top kernel_nlp
add_files "output.cpp" -cflags " -O3 -D XILINX "
add_files -tb "csim.cpp" -cflags " -O3 -D XILINX "
open_solution -flow_target vivado solution
set_part {part}
create_clock -period {freq}MHz -name default
csim_design
close_project
puts "HLS csim completed successfully"
exit
"""
    with open(output_file, "w", encoding="utf-8") as fh:
        fh.write(content)


def generate_tcl_from_profile(
    output_file: str,
    device_profile: dict,
    mode: str = "csynth",
) -> None:
    """Generate a TCL script using parameters from a device profile dict.

    Parameters
    ----------
    output_file : str
        Destination path.
    device_profile : dict
        A device profile as returned by ``DeviceProfile.load()``.
    mode : str
        One of ``"csynth"`` or ``"csim"``.
    """
    part = device_profile["hls_part"]
    freq = str(device_profile.get("default_freq_mhz", 300))
    if mode == "csim":
        generate_csim(output_file, part=part, freq=freq)
    else:
        generate_tcl(output_file, part=part, freq=freq)


def generate_makefile(output_file: str) -> None:
    """Generate a minimal Makefile for HLS csynth and csim."""
    content = """\
.PHONY: hls csim

hls:
\tvitis-run --mode hls --tcl vitis.tcl

csim:
\tvitis-run --mode hls --tcl csim.tcl
"""
    with open(output_file, "w", encoding="utf-8") as fh:
        fh.write(content)