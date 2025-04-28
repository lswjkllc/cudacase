#ifndef CUDA_GEMM_NATIVE_H
#define CUDA_GEMM_NATIVE_H

# include <cuda_runtime.h>

__global__ void matrix_multiply_cuda_naive(int *a, int *b, int *c, int M, int N,int K);

#endif // CUDA_GEMM_NATIVE_H
