"""
reports/v80_attention/golden_compare.py
Comparison script: NumPy reference GEMM vs Vitis HLS csim output.

TODO: Connect to actual csim output by providing:
  - A: input matrix saved by csim.cpp as a binary float32 file (csim_A.bin)
  - B: weight matrix (csim_B.bin)
  - C_hls: output matrix written by csim (csim_C.bin)

Currently this script runs the NumPy reference only.
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np


def reference_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Reference float32 matrix multiplication."""
    return (A @ B).astype(np.float32)


def compare(
    A_path: str,
    B_path: str,
    C_hls_path: str,
    M: int = 197,
    K: int = 768,
    N: int = 768,
    tol: float = 1e-6,
) -> dict:
    """Compare HLS csim output against NumPy reference.

    Parameters
    ----------
    A_path, B_path, C_hls_path : str
        Paths to binary float32 files written by csim.cpp.
    M, K, N : int
        Matrix dimensions (A: M×K, B: K×N, C: M×N).
    tol : float
        Maximum relative error threshold.

    Returns
    -------
    dict with keys: max_rel_error, max_abs_error, pass
    """
    A = np.fromfile(A_path, dtype=np.float32).reshape(M, K)
    B = np.fromfile(B_path, dtype=np.float32).reshape(K, N)
    C_hls = np.fromfile(C_hls_path, dtype=np.float32).reshape(M, N)
    C_ref = reference_matmul(A, B)

    abs_err = np.abs(C_ref - C_hls)
    denom = np.abs(C_ref) + 1e-9
    rel_err = abs_err / denom

    max_abs = float(abs_err.max())
    max_rel = float(rel_err.max())

    result = {
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "pass": max_rel < tol,
        "threshold": tol,
    }
    return result


def demo_reference_only(M: int = 197, K: int = 768, N: int = 768) -> None:
    """Run NumPy reference computation only (no HLS csim files required)."""
    rng = np.random.default_rng(42)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    C = reference_matmul(A, B)

    print(f"Reference GEMM: A({M}×{K}) × B({K}×{N}) → C({M}×{N})")
    print(f"  C[0, :4] = {C[0, :4]}")
    print(f"  |C|_F = {np.linalg.norm(C):.4f}")
    print("[TODO] To compare against HLS csim output, provide --A --B --C_hls flags.")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare HLS csim output against NumPy reference GEMM."
    )
    p.add_argument("--A", default=None, help="Path to input A binary (float32)")
    p.add_argument("--B", default=None, help="Path to weight B binary (float32)")
    p.add_argument("--C_hls", default=None, help="Path to HLS csim output C binary (float32)")
    p.add_argument("--M", type=int, default=197)
    p.add_argument("--K", type=int, default=768)
    p.add_argument("--N", type=int, default=768)
    p.add_argument("--tol", type=float, default=1e-6)
    args = p.parse_args()

    if args.A and args.B and args.C_hls:
        result = compare(args.A, args.B, args.C_hls, args.M, args.K, args.N, args.tol)
        print(f"Max absolute error : {result['max_abs_error']:.2e}")
        print(f"Max relative error : {result['max_rel_error']:.2e}")
        print(f"Threshold          : {result['threshold']:.2e}")
        print(f"PASS               : {result['pass']}")
        sys.exit(0 if result["pass"] else 1)
    else:
        print("[INFO] No csim binary paths provided – running reference-only demo.")
        demo_reference_only(args.M, args.K, args.N)


if __name__ == "__main__":
    main()
