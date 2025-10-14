#include "config.h"
#include "kernels.h"

/**
 * Measure execution time of a GEMM kernel using CUDA events.
 * Automatically times and prints kernel runtime in milliseconds.
 */
void RuntimeEval(void *funcPtr, int mode, const int *A, const int *B, int *C, ...)
{
    cudaEvent_t start, stop;
    float elapsed = 0.0f;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    ((GEMM_TYPE)funcPtr)(A, B, C);  // Run GEMM function

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);

    std::cout << (
        mode == TILE_MODE ? "Tile: " :
        mode == GENERAL_MODE ? "General: " : "Optimized Tile: "
    ) << elapsed << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

/**
 * Pads a 2D matrix (flattened) to make its dimensions multiples of tile sizes.
 * Missing elements are filled with zeros.
 */
int* padding(const int *A, int m, int n, int w, int h)
{
    int pad_m = DIV_ROUND_UP(m, h) * h;
    int pad_n = DIV_ROUND_UP(n, w) * w;
    int *pad_A = (int*)malloc(pad_m * pad_n * sizeof(int));

    for (int i = 0; i < pad_m; ++i)
        for (int j = 0; j < pad_n; ++j)
            pad_A[i * pad_n + j] = (i < m && j < n) ? A[i * n + j] : 0;

    return pad_A;
}

/**
 * Prints a 2D matrix (row-major) with optional stride support for padded arrays.
 */
void Print2DArray(const int *A, int m, int n, ...)
{
    va_list args;
    va_start(args, n);
    int stride = va_arg(args, int);
    va_end(args);

    stride = stride ? stride : n;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j)
            std::cout << A[i * stride + j] << " ";
        std::cout << "\n";
    }
}
int main()
{
    // --- Sanity checks ---
    if (A_n != B_m)  return std::cerr << "Dimension mismatch.\n", 0;
    if (WIDTH > B_n) return std::cerr << "Tile width > B_n.\n", 0;
    if (HEIGHT > A_m) return std::cerr << "Tile height > A_m.\n", 0;
    if (offset > A_n) return std::cerr << "Tile offset > A_n (k-dim).\n", 0;

    // --- Host memory ---
    int C[m * n] = {};
    int *device_A, *device_B, *device_C;

    cudaMalloc(&device_A, A_m * A_n * sizeof(int));
    cudaMalloc(&device_B, B_m * B_n * sizeof(int));
    cudaMalloc(&device_C, m * n * sizeof(int));
    cudaMemcpy(device_A, &**A_host, A_m * A_n * sizeof(int), H2D);
    cudaMemcpy(device_B, &**B_host, B_m * B_n * sizeof(int), H2D);
    cudaMemcpy(device_C, C, m * n * sizeof(int), H2D);

    // --- (1) Naive tiled GEMM ---
    RuntimeEval((void*)Call_Matrix_Mul_Tile, TILE_MODE, device_A, device_B, device_C);
    cudaMemcpy(C, device_C, m * n * sizeof(int), D2H);
    Print2DArray(C, m, n);

    cudaMemset(device_C, 0, m * n * sizeof(int));

    // --- (2) Optimized tiled GEMM ---
    int Pad_m = DIV_ROUND_UP(A_m, HEIGHT) * HEIGHT;
    int Pad_n = DIV_ROUND_UP(B_n, WIDTH) * WIDTH;
    int Pad_C[Pad_m * Pad_n] = {};

    int *Pad_A = padding(&**A_host, A_m, A_n, A_n, HEIGHT);
    int *Pad_B = padding(&**B_host, B_m, B_n, WIDTH, B_m);

    int *device_Pad_A, *device_Pad_B, *device_Pad_C;
    cudaMalloc(&device_Pad_A, A_n * Pad_m * sizeof(int));
    cudaMalloc(&device_Pad_B, B_m * Pad_n * sizeof(int));
    cudaMalloc(&device_Pad_C, Pad_m * Pad_n * sizeof(int));

    cudaMemcpy(device_Pad_A, Pad_A, A_n * Pad_m * sizeof(int), H2D);
    cudaMemcpy(device_Pad_B, Pad_B, B_m * Pad_n * sizeof(int), H2D);
    cudaMemset(device_Pad_C, 0, Pad_m * Pad_n * sizeof(int));

    RuntimeEval((void*)Call_Matrix_Mul_Tile_Cross_SMs, OPTIMIZED_TILE_MODE, device_Pad_A, device_Pad_B, device_Pad_C);
    cudaMemcpy(Pad_C, device_Pad_C, Pad_m * Pad_n * sizeof(int), D2H);

    Print2DArray(Pad_C, m, n, Pad_n);

    free(Pad_A);
    free(Pad_B);
    cudaFree(device_Pad_A);
    cudaFree(device_Pad_B);
    cudaFree(device_Pad_C);

    // --- (3) General GEMM ---

    if (m * n * A_m > 1024)
        return;

    cudaMemset(device_C, 0, m * n * sizeof(int));
    RuntimeEval((void*)Call_Matrix_Mul_General, GENERAL_MODE, device_A, device_B, device_C);
    cudaMemcpy(C, device_C, m * n * sizeof(int), D2H);
    Print2DArray(C, m, n);

    // --- Cleanup ---
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);

    return 0;
}
