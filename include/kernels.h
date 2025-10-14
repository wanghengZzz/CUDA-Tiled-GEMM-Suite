#ifndef KERNELS_H_
#define KERNELS_H_

/* Host-side kernel launcher typedef */
typedef void (*GEMM_TYPE)(const int *A, const int *B, int *C);

/* Host launchers (implemented in kernels.cu) */
void Call_Matrix_Mul_Tile(const int *A, const int *B, int *C);
void Call_Matrix_Mul_Tile_Cross_SMs(const int *A, const int *B, int *C);
void Call_Matrix_Mul_General(const int *A, const int *B, int *C);

#endif // KERNELS_H_
