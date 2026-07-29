# reports/v80_attention

Evidence directory for the AutoHLS_Flow Alveo V80 Transformer Attention benchmark.

## Purpose

This directory provides **transparent, reproducible evidence** for the performance and resource
utilization numbers cited in the main README.  Every claim in the README must trace back to
a file in this directory.

> **Scope**: All numbers are **Vitis HLS C Synthesis Estimates** (Vitis HLS 2025.1).
> Board-level execution on a physical Alveo V80 has not been completed.

## File Index

| File | Description |
|:---|:---|
| [configuration.md](configuration.md) | Exact test parameters: workload, device, toolchain, AMPL/Gurobi solver results |
| [command.txt](command.txt) | Verbatim command used to generate these results |
| [utilization.rpt](utilization.rpt) | Resource utilization section extracted from Vitis HLS csynth.rpt |
| [golden_compare.py](golden_compare.py) | NumPy reference vs HLS csim output comparison script (stub – connect to actual csim output) |

## Benchmark Configuration Summary

| Parameter | Value |
|:---|:---|
| **Workload** | DeiT/ViT Transformer Attention Block – 4× MatMul (197×768×768, float32) |
| **Device** | AMD Alveo V80 (`xcv80-lsva4737-2MHP-e-S`) |
| **Toolchain** | Vitis HLS 2025.1 (Build 6135595, May 21 2025) |
| **Solver** | AMPL + Gurobi 13.0.0 |
| **Target Clock** | 300 MHz (3.33 ns period) |
| **Estimated Fmax** | 411.37 MHz (2.431 ns achieved) |
| **Estimated Latency** | 3,651,775–3,657,121 cycles → **12.16 ms** at 300 MHz |

## Claimed Numbers in README and Their Source

| README Claim | Source File | Status |
|:---|:---|:---|
| Latency 12.16 ms | configuration.md §5, utilization.rpt | ✅ C Synthesis Estimate |
| Fmax 411 MHz | configuration.md §5, utilization.rpt | ✅ C Synthesis Estimate |
| DSP 1.1% | utilization.rpt | ✅ C Synthesis Estimate |
| LUT 26.8% | utilization.rpt | ✅ C Synthesis Estimate |
| BRAM 53.7% | utilization.rpt | ✅ C Synthesis Estimate |
| CSIM accuracy < 1e-6 | configuration.md §5 | ✅ CSIM verified |

## Reproducing the Results

See [command.txt](command.txt) for the exact command.

Requires: AMPL + Gurobi 13.0.0, Vitis HLS 2025.1, Python 3.10+, `amplpy`.
