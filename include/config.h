#ifndef CONFIG_H_
#define CONFIG_H_

#include <cuda_runtime.h>
#include <cstdarg>
#include <iostream>
#include <cstdlib>

/* -------------------------------------------------
 * Macros and constants
 * ------------------------------------------------- */
#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost
#define TILE_MODE 0
#define OPTIMIZED_TILE_MODE 1
#define GENERAL_MODE 2
#define DIV_ROUND_UP __KERNEL_DIV_ROUND_UP
#define __KERNEL_DIV_ROUND_UP(n, d) (((n) + (d) - 1) / (d))
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

#ifndef INCLUDE_MATRIX_DATA
#define INCLUDE_MATRIX_DATA
#include MATRIX_FILE
#endif

/* Utility function declarations */
int* padding(const int *A, int m, int n, int w, int h);
void Printf2dArray(const int *A, int m, int n, ...);

#endif // CONFIG_H_
