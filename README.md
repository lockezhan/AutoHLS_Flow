# AutoHLS_Flow: Automatic HLS Optimization and Code Generation Framework

AutoHLS_Flow is a holistic, optimization-driven toolchain for automatic high-level synthesis (HLS) code generation for FPGA accelerators. It supports both High-Level Synthesis from affine C/C++ kernels and direct parsing & mapping from ONNX neural network models, leveraging polyhedral optimization, non-linear programming (AMPL/Gurobi), and dataflow architecture to generate highly efficient HLS-C++ pipelines and host code for AMD Vitis HLS.

---

## 🏗️ Framework Architecture

The overall compilation and optimization pipeline of AutoHLS_Flow is illustrated below:

```mermaid
flowchart TD
    subgraph Input ["1. Frontends & Parsing"]
        A1[ONNX Models .onnx] -->|onnx_frontend.py| AST[Internal AST & Affine Loops]
        A2[Affine C/C++ Kernels] -->|parser.py / extract.py| AST
    end

    subgraph Optimization ["2. Mathematical Optimization & Polyhedral Analysis"]
        AST -->|iscc.py / polyhedro.py| Poly[Polyhedral Loop Bounds & Dependencies]
        Poly -->|memoryBound.py / analysis.py| NLP[AMPL NLP Mathematical Model]
        NLP -->|Gurobi / AMPL Solver| Sol[Optimal Tiling & Unroll Factors]
    end

    subgraph Partitioning ["3. Graph & Resource Partitioning"]
        Sol -->|splitKernel.py| Dataflow[Dataflow Graph & Fused Task Partitioning]
        Dataflow -->|SLR Mapper| SLR[SLR/CU Resource Allocation]
    end

    subgraph CodeGen ["4. HLS C++ & Host Code Generation"]
        SLR -->|code_generation_dataflow.py| HLS[HLS Top-level output.cpp & slrX.cpp]
        SLR -->|host_generation| Host[Host Control Code & XRT Wrappers]
        HLS --> PingPong[Ping-Pong Buffers & Stream FIFOs Generation]
    end

    subgraph Synthesis ["5. Vitis HLS & Hardware Deployment"]
        PingPong --> Vitis[AMD Vitis HLS / AVED 25.1]
        Vitis --> CSIM[C Simulation / Accuracy Check]
        Vitis --> Synth[C Synthesis & IP Integration]
        Synth --> PDI[Bitstream / PDI Generation]
    end
```

### Architecture Highlights
- **Polyhedral Loop Scheduling**: Automatically extracts loop dependencies and static iteration bounds.
- **NLP-based Resource Allocation**: Formulates hardware optimization into Non-Linear Programming models solved via AMPL & Gurobi for optimal tiling ($TC$) and unrolling factors ($UF$).
- **Fused Task (FT) Dataflow Engine**: Transforms nested loops into fine-grained Fused Tasks communicating via HLS AXI-Stream FIFOs.
- **Ping-Pong Buffers & Memory Hiding**: Decouples global DMA transfers from local tile computations to hide memory latencies.

---

## 🧩 Supported ONNX Operators & Limitations

AutoHLS_Flow features a custom ONNX frontend (`onnx_frontend.py`) designed to map deep learning operators directly to hardware dataflow pipelines.

### 1. Supported ONNX Operators

| Operator Category | ONNX Operator | Supported Shapes & Dimensions | Hardware Mapping Strategy |
| :--- | :--- | :--- | :--- |
| **Matrix Operations** | `MatMul`, `Gemm` | 2D/3D/4D (e.g., `[B, M, K] x [B, K, N]`) | Tiled Systolic/Dataflow MatMul with Ping-Pong Buffers |
| **Element-wise Math** | `Add`, `Sub`, `Mul`, `Div` | Any shape matching operand bounds | Pipelined element-wise streaming compute units |
| **Activations** | `Relu` | Arbitrary Tensor shapes | Fully unrolled / pipelining inner compute loops |
| **Normalization & Reduction** | `LayerNormalization`, `Softmax` | 2D/3D Tensors (Layer-dim reduction) | Intra-tile reduction trees with intermediate FIFOs |

### 2. Constraints & Limitations

- **Static Bounds Requirement**: Loop bounds and array dimensions must be statically determinable at compile time for `#pragma HLS ARRAY_PARTITION` and buffer allocation. Dynamic batch sizes are defaulted to 1 or resolved via `--update_shape`.
- **Affine Access Patterns**: Memory access indices must be affine combinations of loop iterators (e.g., `A[i][k]`, `B[k][j]`). Non-affine indirect accesses (e.g., `A[B[i]]`) are not currently supported by the polyhedral scheduler.
- **Data Types**: Native support for `float` (32-bit floating point) and `double`. High-throughput fixed-point (`ap_int`, `ap_fixed`) generation is supported via C++ template specialization.
- **Graph Topology**: Currently optimizes feed-forward DAG (Directed Acyclic Graph) topologies typical in Vision Transformers (e.g., DeiT, ViT) and GPT attention blocks.

---

## 🚀 Complete End-to-End Example

Below is a complete walk-through from an input ONNX model to a fully generated Vitis HLS project ready for simulation and synthesis.

### Step 1: Run AutoHLS_Flow Command

To generate an HLS dataflow project targeting **AMD Versal HBM / Alveo V80** from an ONNX attention model:

```bash
python main.py \
  --onnx_file onnx_files/deit_model.onnx \
  --device Alveo_V80 \
  --SLR 3 \
  --DSP 1440 \
  --MAX_BUFFER_SIZE 512 \
  --ON_CHIP_MEM_SIZE 8192 \
  --MAX_UF 32 \
  --code_generation \
  --vitis \
  --csim \
  --folder hls_output_demo
```

### Step 2: Generated HLS Project Directory Structure

Upon completion, `hls_output_demo/` contains all top-level HLS C++ files, multi-SLR partition files, Host XRT wrappers, and TCL synthesis scripts:

```text
hls_output_demo/
├── Makefile                      # Top-level build script for Vitis HLS
├── build.tcl                     # Vivado/Vitis synthesis driver
├── hls_config_slr0.cfg           # Vitis configuration file
├── hls_run.sh                    # One-click execution script
├── nlp.mod                       # Generated AMPL mathematical optimization model
├── nlp.log                       # Solver logs & optimized tiling factors
├── src/
│   ├── output.cpp                # Top-level HLS Dataflow C++ code
│   ├── output.h                  # Top-level headers, struct & array declarations
│   ├── output_2.h                # Internal stream & FIFO definitions
│   ├── slr0.cpp                  # Kernel compute logic mapped to SLR 0
│   ├── slr1.cpp                  # Kernel compute logic mapped to SLR 1
│   ├── slr2.cpp                  # Kernel compute logic mapped to SLR 2
│   ├── host.cpp                  # OpenCL / XRT Host host execution code
│   ├── csim.tcl                  # TCL script for C Simulation
│   ├── vitis.tcl                 # TCL script for C Synthesis
│   ├── xcl2.cpp / xcl2.hpp       # Xilinx OpenCL helper library
└── tcl_scripts/                  # Secondary placement & physical opt TCL scripts
```

### Step 3: Run C Simulation & Synthesis

Navigate into the generated project folder and run Vitis HLS:

```bash
cd hls_output_demo
bash hls_run.sh
```

---

## 📊 CSIM & C Synthesis Accuracy & Performance Results

We evaluated AutoHLS_Flow on Transformer Attention blocks (DeiT/GPT) mapped onto **AMD Versal HBM / Alveo V80 (xcv80-lsva4737-2MHP-e-S)** using **AMD Vitis HLS 2025.1**.

### 1. Correctness & CSIM Validation
- **Functional Accuracy**: C Simulation (`csim.tcl`) was verified against NumPy / PyTorch golden reference outputs. Maximum relative error observed was $< 10^{-6}$ for float32 dataflows.
- **FIFO Deadlock Freedom**: Verified through HLS Dataflow static channel analysis and CSIM runtime execution; all stream channels operate without stalling or deadlocks.

### 2. C Synthesis & Performance Summary

| Metric | Result / Measurement | Notes / Target |
| :--- | :--- | :--- |
| **Target Device** | AMD Versal Alveo V80 (`xcv80-lsva4737`) | Versal Architecture |
| **Target Clock Period** | `3.33 ns` (300 MHz) | Constraint Target |
| **Estimated Clock Period** | **`2.431 ns`** | **`0.90 ns` Clock Margin** |
| **Estimated Max Freq ($F_{max}$)** | **`411.37 MHz`** | Exceeds Target Frequency |
| **Execution Latency (Cycles)** | `3,651,775` ~ `3,657,121` cycles | Dataflow Pipeline Latency |
| **Absolute Execution Time** | **`12.16 ms`** | Full Attention Graph Run |

### 3. Resource Utilization Breakdown

| Resource Type | Used | Total Available | Utilization (%) |
| :--- | :--- | :--- | :--- |
| **BRAM_18K** | 4,020 | 7,482 | **53.7%** |
| **DSP** | 128 | 10,848 | **1.1%** |
| **Flip-Flops (FF)** | 366,375 | 5,148,416 | **7.1%** |
| **Look-Up Tables (LUT)**| 691,055 | 2,574,208 | **26.8%** |
| **URAM** | 0 | 1,925 | 0% (Configurable) |

---

## ✨ Key Features

- Direct parsing and mapping of **ONNX models** to optimized HLS-C++ pipelines.
- Support for **affine C/C++ kernels** with static loop bounds.
- Automatic **loop scheduling**, **pragmas insertion**, and **code generation**.
- Integration with **AMPL** for **Nonlinear Programming (NLP)**-based resource allocation.
- Generation of optimized **HLS-C++** and **host code**.
- Simulation and synthesis with **AMD Vitis HLS**.

---

## 🚀 Quick Start & Environment Setup

### 1. Requirements

- Python 3.8+
- AMPL (configured in `main.py`)
- Clang-format (for formatting the output code)
- AMD Vitis HLS (for synthesis and CSIM)

To run the framework seamlessly with PoCC, ISCC, AMPL, and Gurobi dependencies, we provide a pre-configured Docker image.

**1. Pull the Docker Image:**

```bash
docker pull ryanzhang511/autohls_flow_image:latest
```

**2. Start the Docker Container:**

You must map your local directory to the container and provide your AMPL license UUID via an environment variable.

```bash
docker run -it -d \
  --name autohls_flow_container \
  -v /path/to/your/AutoHLS_Flow:/AutoHLS_Flow \
  -e AMPL_LIC_UUID="<your-ampl-license-uuid>" \
  ryanzhang511/autohls_flow_image:latest \
  /bin/bash
```

**3. Enter the Container:**

```bash
docker exec -it autohls_flow_container /bin/bash
cd /AutoHLS_Flow
```

---

## ⚙️ Command-Line Arguments

| Argument                 | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `--file`                 | Path to the input C/C++ kernel                                              |
| `--onnx_file`            | Path to the input ONNX neural network model file                            |
| `--folder`               | Output folder for generated code and reports (default: `hls_output`)        |
| `--device`               | Target device profile (`Alveo_V80`, `AC7t1500`, etc.)                       |
| `--name_function`        | Kernel function name (default: `kernel_nlp`)                               |
| `--SLR`                  | Number of available Super Logic Regions                                    |
| `--DSP`                  | Total number of available DSP slices                                       |
| `--BRAM`, `--FF`, `--LUT`| FPGA resource budgets (optional)                                           |
| `--MAX_BUFFER_SIZE`      | Maximum allowed on-chip buffer size per array                             |
| `--ON_CHIP_MEM_SIZE`     | Total available on-chip memory                                             |
| `--MAX_UF`               | Maximum loop unrolling factor                                              |
| `--reuse_nlp`            | Use previously computed NLP results                                        |
| `--vitis`                | Enable synthesis using AMD Vitis HLS                                       |
| `--csim`                 | Enable C simulation with Vitis HLS                                         |
| `--code_generation`      | Enable output of HLS-C++ and host code                                     |
| `--graph_partitioning`   | Enable graph partitioning across SLRs or compute units                     |
| `--no_distribution`      | Disable ISCC-based loop distribution                                       |
| `--update_shape`         | Automatically update shape-related constraints                             |
| `--ap_multiple_burst`    | Enable multiple AXI burst access inference                                 |
| `--cyclic_buffer`        | Use cyclic buffering strategy for data reuse                               |
| `--node_limit`           | Limit number of ONNX nodes to compile                                      |
| `--has_uram`             | Force enabling URAM storage binding for large arrays                       |
| `--verbose`              | Print detailed information during execution                                |
| `--debug`                | Enable debug mode                                                          |
