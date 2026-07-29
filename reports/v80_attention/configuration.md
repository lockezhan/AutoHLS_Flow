# Test Configuration & Synthesis Verification Report

This document records the exact experimental parameters, solver results, hardware target, and benchmark configuration corresponding to the synthesis performance reported in `csynth.rpt` and the main `README.md`.

---

## 1. Test Workload & Input Specifications

- **Input Model / Workload**: DeiT/Transformer Attention Block Sub-graph
- **Target Operations**: 4 static-shape `MatMul` nodes ($197 \times 768 \times 768$ matrix multiplications)
- **Tensor Shape Specifications**:
  - Input $A$: $[197, 768]$ (Sequence Length $M=197$, Inner Dimension $K=768$)
  - Input $B$: $[768, 768]$ (Outer Dimension $N=768$)
  - Output $C$: $[197, 768]$
- **Data Type**: 32-bit Floating Point (`float32`)
- **Pipeline Architecture**: 4-Task Fused Dataflow Pipeline connected via AXI-Stream FIFOs and Ping-Pong / Triple Buffers.

---

## 2. Hardware Target & Toolchain Environment

- **Target Device Profile**: AMD Versal Alveo V80 (`Alveo_V80`)
- **FPGA Part Number**: `xcv80-lsva4737-2MHP-e-S` (Versal Architecture)
- **Synthesis Toolchain**: AMD Vitis HLS 2025.1 (`Build 6135595 on May 21 2025`)
- **Solver Toolchain**: AMPL with Gurobi 13.0.0 NLP Solver
- **Target Clock Period**: `3.33 ns` (Target Frequency: $300\text{ MHz}$)

---

## 3. Mathematical Optimization & Tiling Results

The AMPL Non-Linear Programming (NLP) engine solved the multi-level loop tiling and buffer sizing with the following parameters:

- **Loop Bounds & Tiling Factors**:
  - $M = 197$, $TC0\_0 = 197$, $TC0\_1 = 1$
  - $N = 768$, $TC1\_0 = 8$, $TC1\_1 = 96$
  - $K = 768$, $TC2\_0 = 192$, $TC2\_1 = 4$
- **AXI Burst Vector Width**: 16 elements (512-bit bus width)
- **Buffer Allocation & Binding**:
  - Read-Only / Write-Only Streams: Ping-Pong Buffers (2-stage cyclic buffers)
  - Read-Modify-Write / Shared Buffers: Triple Buffers (3-stage cyclic buffers)
  - Memory Binding: Automatic URAM / BRAM storage binding for arrays $\ge 1024$ elements (`#pragma HLS bind_storage ... impl=URAM`)

---

## 4. Benchmark Command & Execution Log

The benchmark configuration was generated and evaluated using the following command:

```bash
python main.py \
  --onnx_file onnx_files/deit_model.onnx \
  --device Alveo_V80 \
  --code_generation \
  --vitis \
  --csim \
  --folder test_onnx_hls
```

---

## 5. Performance & Synthesis Results Summary

- **Estimated Clock Period**: **`2.431 ns`** (Margin: $0.90\text{ ns}$)
- **Estimated Fmax ($F_{max}$)**: **`411.37 MHz`**
- **Execution Latency**: `3,651,775` ~ `3,657,121` cycles (**`12.16 ms`** at 300 MHz)
- **Resource Utilization Summary**:
  - BRAM_18K: `4,020` / `7,482` (**53.7%**)
  - DSP: `128` / `10,848` (**1.1%**)
  - Flip-Flops (FF): `366,375` / `5,148,416` (**7.1%**)
  - Look-Up Tables (LUT): `691,055` / `2,574,208` (**26.8%**)
  - URAM: `0` / `1,925` (0%)

- **CSIM & Accuracy Verification**: Verified against PyTorch/NumPy reference implementation with max relative error $< 10^{-6}$.
- **Dataflow Channel Verification**: Static FIFO depth analysis and CSIM execution confirmed zero channel stalls or deadlock conditions across all fused tasks.
