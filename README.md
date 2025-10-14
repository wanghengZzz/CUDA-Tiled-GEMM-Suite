# CUDA Tiled GEMM Suite

A high-performance CUDA implementation of **General Matrix Multiplication (GEMM)** that demonstrates:
- **Basic tiled multiplication**
- **Optimized tiled kernel spanning multiple SMs**
- **General (non-tiled) reference implementation**

> This project was tested on an **NVIDIA GeForce RTX 3060 Ti (8 GB VRAM)** with **CUDA 12.2** and **Driver 535.230.02**, using compute architecture **sm_80**.  
> The design supports **variable tile sizes (`WIDTH`, `HEIGHT`)** and **flexible depth tiles (`k_offset`)** — they **do not need to evenly divide** the matrix dimensions.

---

## 🔧 Key Features

- **Shared-memory tiling** for sub-blocks of matrices A and B  
- **Coalesced global memory loads** via strided thread access patterns  
- **Register-level accumulation** to minimize shared/global memory traffic  
- **No atomic operations** in the optimized version — each thread computes a unique `C` element  
- **No branch divergence** — the optimized tiled kernel avoids all `if / else` conditions for full warp efficiency  
- **Supports non-divisible matrix sizes** (handles remainder tiles gracefully)  
- Optional **matrix padding** improves coalescing and simplifies edge-case handling  

---

## ⚙️ Build & Run

### Requirements
- CUDA 11 or later (tested on CUDA 12.2)
- NVIDIA GPU with SM 70 or newer (tested on RTX 3060 Ti)
- GNU Make + g++
- Python 3.6+

### Commands
```bash
make          # Compile all sources
make run      # Execute the demo binary (bin/matmul)
make clean    # Remove build/ and bin/ folders
```

## 🧮 Generating Custom Matrices

You can easily generate custom random matrices for testing using the provided helper script:

```shell
./gen_matrix.sh
```

Explanation of each parameter used in the `gen_matrix.sh` execution command:

```python

"""
Generate random 2D matrices for CUDA Tiled GEMM experiments.

This script creates a text file (`matrix_data.txt`) containing two matrices (A and B)
and tiling configuration parameters. The generated data will later be compiled into
a header file (`matrix_data.h`) for CUDA programs.

Command-line usage example:
---------------------------
python3 tools/gen_matrix.py \
  --WIDTH 23 \                # Number of rows in matrix A
  --HEIGHT 17 \               # Number of columns in matrix B
  --OFFSET 19 \               # Number of columns in A (and rows in B)
  --OFFSET_K 3 \              # k-offset, used as the tile depth (C[i][j] = A[i][k] * B[k][j])
  --TILE_WIDTH 4 \            # Tile width (threads per block in X direction)
  --TILE_HEIGHT 3 \           # Tile height (threads per block in Y direction)
  --MAX_VALUE 10 \            # Maximum value for random number generation
  --MIN_VALUE -10 \           # Minimum value for random number generation
  --OUT_PTH data/matrix_data.txt  # Output path of the generated matrix data file

Notes:
------
- The script produces integer-valued matrices within the specified range.
- WIDTH, HEIGHT, and OFFSET define the dimensions for matrix multiplication:
      A: (WIDTH × OFFSET)
      B: (OFFSET × HEIGHT)
      C: (WIDTH × HEIGHT)
- TILE_WIDTH, TILE_HEIGHT, and OFFSET_K can be arbitrary and do NOT need to divide
  the matrix dimensions evenly.
"""

```

You can specify any matrix file at runtime via `make`

```bash
make clean
make run MATRIX_TXT=data/your_matrix_data.txt

```

This will automatically invoke the Python generator to produce
`include/matrix_data.h` before compilation, embedding the chosen matrix.


## 🧠 Notes
- Tile width (WIDTH), tile height (HEIGHT), and k_offset are fully configurable.
They do not need to evenly divide the matrix dimensions.
- All kernels automatically handle non-divisible remainders and optional padding.
- The optimized tiled kernel removes all conditional branches for higher occupancy and warp-level efficiency.