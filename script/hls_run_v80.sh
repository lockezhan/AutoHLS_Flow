#!/usr/bin/env bash
# hls_run_v80.sh – Stub for Alveo V80 builds.
#
# Alveo V80 does NOT use the Vitis OpenCL / v++ / xclbin flow.
# Physical deployment on V80 is handled by the SLASH/VRT backend.
#
# To deploy AutoHLS_Flow-generated kernels on V80:
#
#   Step 1: Generate HLS C++ code
#     python main.py --onnx_file ... --device Alveo_V80 --code_generation [--vitis] [--csim] --folder hls_output
#
#   Step 2: Export SLASH project bundle
#     python integrations/slash/export_to_slash.py \
#       --hls-project hls_output \
#       --slash-root /path/to/SLASH \
#       --project-name my_project \
#       --output-dir slash_projects/my_project
#
#   Step 3: Build and deploy via SLASH CMake flow
#     cd slash_projects/my_project
#     bash run_v80.sh <BDF> /path/to/SLASH
#
# Prerequisites (one-time, per machine):
#   - SLASH installed and initialized (AVED 25.1)
#   - Abstract shell DCP built (see SLASH linker docs)
#   - V80 card flashed with base platform image
#
# See integrations/slash/README.md for the full guide.

set -euo pipefail
echo "[ERROR] V80 builds use the SLASH/VRT CMake flow, not hls_run.sh."
echo "        Please use integrations/slash/export_to_slash.py to generate"
echo "        a SLASH project, then follow the steps in integrations/slash/README.md."
exit 1
