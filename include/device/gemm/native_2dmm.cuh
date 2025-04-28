#ifndef CUDA_GEMM_NATIVE_H
#define CUDA_GEMM_NATIVE_H

# include <cuda_runtime.h>

__global__ void SgemmWithNative(int *a, int *b, int *c, int M, int N,int K);

#endif // CUDA_GEMM_NATIVE_H
