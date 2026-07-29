catch {::common::set_param -quiet hls.xocc.mode csynth};

# NOTE: This static template is only a reference.
# The actual part and frequency are set by code_gen/write_tcl.py using
# the device profile (device_profiles/alveo_v80.json or similar).
# Do NOT hardcode a device part string here.

open_project kernel_nlp
set_top kernel_nlp

add_files "output.cpp"
open_solution -flow_target vivado solution
# set_part is written by code_gen/write_tcl.generate_tcl_from_profile()

create_clock -period 300MHz -name default

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