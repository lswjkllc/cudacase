# include <cuda_runtime.h>

__global__ void matrix_multiply_cuda_naive(float *a, float *b, float *c, int M, int N,int K);