#include "config.h"
#include "kernels.h"

/* Shared memory buffer */
extern __shared__ int S[];

/* -------------------------------------------------
 * 1. Basic tiled matrix multiplication kernel
 * ------------------------------------------------- */
__global__ void Matrix_Mul_Tile(const int *A, const int *B, int * __restrict__ C, int w, int h, int base_i, int base_j, int base_k, int k_offset)
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    int k = threadIdx.z;

    S[i * k_offset + k]       = A[(i + base_i) * A_n + (k + base_k)];
    S[(h + j) * k_offset + k] = B[(k + base_k) * B_n + (j + base_j)];

    __syncthreads();

    atomicAdd(&C[(i + base_i) * B_n + (j + base_j)],
              S[i * k_offset + k] * S[(h + j) * k_offset + k]);
}

/* -------------------------------------------------
 * 2. Optimized tiled kernel across multiple SMs
 * -------------------------------------------------
 * Optimizations:
 * - Coalesced global memory access using strided thread loading
 * - Shared-memory tiling for A and B
 * - Register-level accumulation for C
 * - Synchronization barrier to avoid race conditions
 * - Avoid atomic operations by assigning one thread per C element
 *
 * Note:
 * - This kernel **requires matrix padding** so that both A and B
 *   dimensions are multiples of the tile size (HEIGHT × WIDTH).
 *   Padding ensures that all threads and warps operate on valid
 *   memory addresses and maintain coalesced memory access patterns.
 */

__global__ void Matrix_Mul_Tile_Cross_SMs_K(const int *A, const int *B, int * __restrict__ C, int w, int h, int Pad_B_n, int base_k, int k_offset)
{
    int bi = blockIdx.x;
    int bj = blockIdx.y;
    int i = threadIdx.x;
    int j = threadIdx.y;

    int tot = 0;

    for (int k = j; k < k_offset; k += blockDim.y)
        S[i * k_offset + k] = A[(i + bi * h) * A_n + (k + base_k)];

    for (int k = i; k < k_offset; k += blockDim.x)
        S[(h + j) * k_offset + k] = B[(k + base_k) * Pad_B_n + (j + bj * w)];

    __syncthreads();

    for (int k = 0; k < k_offset; ++k)
        tot += S[i * k_offset + k] * S[(h + j) * k_offset + k];

    C[(i + bi * h) * Pad_B_n + (j + bj * w)] += tot;
}

/* -------------------------------------------------
 * 3. General GEMM (non-tiled reference implementation)
 * ------------------------------------------------- */
__global__ void Matrix_Mul_General_K(const int *A, const int *B, int * __restrict__ C)
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    int tot = 0;

    for (int k = 0; k < A_n; ++k)
        tot += A[i * A_n + k] * B[k * B_n + j];

    C[i * n + j] = tot;
}

/* -------------------------------------------------
 * Host launchers
 * ------------------------------------------------- */
void Call_Matrix_Mul_Tile_Cross_SMs(const int *A, const int *B, int *C)
{
    int sharedMemSize;
    int k_offset_local;
    int round_up_h   = DIV_ROUND_UP(m, HEIGHT);
    int round_up_w   = DIV_ROUND_UP(n, WIDTH);
    int round_up_B_n = round_up_w * WIDTH;

    dim3 BlocksPerGrid(round_up_h, round_up_w);

    for (int k = 0; k < A_n; k += offset) {
        k_offset_local = min(A_n - k, offset);
        dim3 ThreadsPerBlock(HEIGHT, WIDTH);
        k_offset_pad = k_offset_local + (k_offset_local & (32 - 1) ? 0 : 1);
        sharedMemSize = ((HEIGHT + WIDTH) * k_offset_pad) * sizeof(int);
        Matrix_Mul_Tile_Cross_SMs_K<<<BlocksPerGrid, ThreadsPerBlock, sharedMemSize>>>(
            A, B, C, WIDTH, HEIGHT, round_up_B_n, k, k_offset_pad
        );
    }
}

void Call_Matrix_Mul_Tile(const int *A, const int *B, int *C)
{
    int sharedMemSize;
    int w, h;
    int k_offset_local;
    int k_offset_pad;

    for (int k = 0; k < A_n; k += offset) {
        for (int bi = 0; bi < m; bi += HEIGHT) {
            for (int bj = 0; bj < n; bj += WIDTH) {
                w = min(WIDTH,  n - bj);
                h = min(HEIGHT, m - bi);
                k_offset_local = min(A_n - k, offset);
                k_offset_pad = k_offset_local + (k_offset_local & (32 - 1) ? 0 : 1);
                dim3 ThreadsPerBlock(h, w, k_offset_local);
                sharedMemSize = ((h + w) * k_offset_pad) * sizeof(int);

                Matrix_Mul_Tile<<<1, ThreadsPerBlock, sharedMemSize>>>(
                    A, B, C, w, h, bi, bj, k, k_offset_pad
                );
            }
        }
    }
}

void Call_Matrix_Mul_General(const int *A, const int *B, int *C)
{
    dim3 ThreadsPerBlock(m, n);
    Matrix_Mul_General_K<<<1, ThreadsPerBlock>>>(A, B, C);
}
