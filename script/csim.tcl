catch {::common::set_param -quiet hls.xocc.mode csynth};

# NOTE: This static template is only a reference.
# The actual part and frequency are set by code_gen/write_tcl.py using
# the device profile (device_profiles/alveo_v80.json or similar).
# Do NOT hardcode a device part string here.

open_project csim.prj
set_top kernel_nlp
add_files "output.cpp" -cflags " -O3 -D XILINX "
add_files -tb "csim.cpp" -cflags " -O3 -D XILINX "
open_solution -flow_target vivado solution
# set_part is written by code_gen/write_tcl.generate_tcl_from_profile()

create_clock -period 300MHz -name default
csim_design
close_project
puts "HLS csim completed successfully"
exit