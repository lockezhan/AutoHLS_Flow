# SLASH/VRT Deployment Guide for AutoHLS_Flow

## Responsibility Boundary

| Layer | Tool | Scope |
|:---|:---|:---|
| **Frontend & Optimization** | AutoHLS_Flow | ONNX/C++ parsing → NLP → HLS C++ generation |
| **C Simulation** | Vitis HLS | Functional verification (no hardware) |
| **C Synthesis** | Vitis HLS | Resource/timing estimates (no hardware) |
| **Physical Deployment** | **SLASH/VRT** | Segmented configuration, VRTBIN, board execution |

AutoHLS_Flow generates HLS C++ artifacts. It does **not** program V80 boards.  
Physical deployment requires [SLASH](https://github.com/hpc-aulmamei/SLASH.git) and the AVED platform.

---

## Prerequisites (one-time per machine)

1. **Install AVED / SLASH runtime** per the [SLASH README](https://github.com/hpc-aulmamei/SLASH.git)
2. **Build the abstract shell DCP** (required once per SLASH installation):
   ```bash
   cd /path/to/SLASH/linker/src
   python3 main.py install --build-dir /path/to/SLASH/linker/resources/abstract_shell_build
   ```
   If this fails with missing IPs:
   ```bash
   cd /path/to/SLASH/linker/resources/base/iprepo
   make
   ```
3. **Initialize the V80 card** with the base platform image (see SLASH docs)
4. **Set environment**: `source /opt/xilinx/xrt/setup.sh` and Vitis HLS 2025.1

---

## End-to-End Example: Transformer Attention on V80

### Step 1 — Optimize and Generate HLS C++

```bash
python main.py \
  --onnx_file onnx_files/deit_model.onnx \
  --device Alveo_V80 \
  --node_limit 4 \
  --code_generation \
  --vitis \
  --csim \
  --folder hls_output_attention
```

**Generated output** (`hls_output_attention/src/`):
- `output.cpp` — fused dataflow kernel (all SLR partitions)
- `slr0.cpp`, `slr1.cpp`, `slr2.cpp` — SLR-partitioned variants
- `host.cpp` — OpenCL host code (legacy; not used for V80)
- `csim.cpp` — testbench for Vitis HLS csim
- `vitis.tcl`, `csim.tcl` — device-aware TCL scripts

### Step 2 — Export SLASH Project Bundle

```bash
python integrations/slash/export_to_slash.py \
  --hls-project hls_output_attention \
  --slash-root /path/to/SLASH \
  --project-name autohls_attention \
  --output-dir slash_projects/autohls_attention \
  --verbose
```

Or via `main.py`:
```bash
python main.py \
  --onnx_file onnx_files/deit_model.onnx \
  --device Alveo_V80 \
  --code_generation \
  --backend slash \
  --slash_root /path/to/SLASH \
  --project_name autohls_attention \
  --folder hls_output_attention
```

**Generated output** (`slash_projects/autohls_attention/`):
```
autohls_attention/
├── output.cpp           # HLS kernel source (copy from AutoHLS_Flow)
├── slr0.cpp             # Per-SLR sources (if multi-SLR)
├── slr1.cpp
├── slr2.cpp
├── output.h             # Kernel header
├── output.cfg           # SLASH HLS compilation config
├── config.cfg           # SLASH connectivity (nk=, sp=, slr=, stream_connect=)
├── slr_constraints.tcl  # SLR pblock constraints (pre_synth hook)
├── CMakeLists.txt       # Build integration
├── host.cpp             # VRT runtime host code
├── run_v80.sh           # One-shot build & run script
└── autohls_flow_manifest.json  # Provenance and tiling metadata
```

### Step 3 — Build VRTBIN and Deploy

```bash
cd slash_projects/autohls_attention
bash run_v80.sh <BDF> /path/to/SLASH
```

This runs:
1. `cmake -B build -S . -G Ninja -DSLASH_USE_REPO=ON ...`
2. `cmake --build build --target hls_kernel_nlp` (HLS IP catalog build)
3. `cmake --build build --target autohls_attention_hw` (VRTBIN link)
4. `v80-smi program build/autohls_attention_hw.vbin -d <BDF>` (flash card)
5. `./build/host <BDF> build/autohls_attention_hw.vbin` (run application)

---

## Dry-Run (Validation Only)

```bash
python integrations/slash/export_to_slash.py \
  --hls-project hls_output_attention \
  --project-name autohls_attention \
  --output-dir slash_projects/autohls_attention \
  --dry-run
```

Runs all pre-flight checks without writing any files.

---

## Pre-flight Checks

The exporter validates the following before writing any files:

| Check | Expected |
|:---|:---|
| HLS project directory exists | `hls_project/` |
| Kernel header present | `src/output.h` |
| At least one kernel source | `src/output.cpp` or `src/slr0.cpp` |
| NLP results available | `nlp.log` |
| SLASH root exists (if given) | valid path |
| Abstract shell DCP present | `SLASH/linker/resources/abstract_shell/abs_shell_slash.dcp` |
| Kernel name not a reserved word | Verilog/VHDL identifiers |

---

## Manifest Schema

`autohls_flow_manifest.json` records the full provenance:

```json
{
  "schema_version": "1.0",
  "generation_timestamp": "...",
  "autohls_flow_version": "<git SHA>",
  "slash_bridge_version": "<git SHA>",
  "project_name": "autohls_attention",
  "kernel_name": "kernel_nlp",
  "kernel_count": 3,
  "kernel_functions": ["kernel_nlp_slr0", "kernel_nlp_slr1", "kernel_nlp_slr2"],
  "target_freq_mhz": 300,
  "hls_part": "xcv80-lsva4737-2MHP-e-S",
  "slr_mapping": {"slr0": "kernel_nlp_slr0_0", ...},
  "nlp_tiling": {"TC0_0": 197, "TC1_0": 8, ...},
  "synthesis_status": "C Synthesis Estimate. Board-level execution not yet verified."
}
```

---

## Known Limitations

- AutoHLS_Flow only supports static-shape 2D MatMul (prototype). Higher-rank tensors are decomposed into 2D.
- Physical V80 board results have not been published. All performance numbers in `reports/` are Vitis HLS C Synthesis estimates.
- The SLASH export adapter does not modify or verify SLASH source code.
