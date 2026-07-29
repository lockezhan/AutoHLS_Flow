import numpy as np
import sys

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <reference.npy> <hls.npy>")
        sys.exit(1)
        
    ref = np.load(sys.argv[1])
    hls = np.load(sys.argv[2])
    
    if ref.shape != hls.shape:
        print(f"Shape mismatch: {ref.shape} vs {hls.shape}")
        sys.exit(1)
        
    err = np.abs(ref - hls).max()
    print(f"Max Absolute Error: {err}")
    if err < 1e-6:
        print("Validation PASSED.")
    else:
        print("Validation FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    main()
