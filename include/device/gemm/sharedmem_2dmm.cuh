# include <cuda_runtime.h>

# define BLOCK_SIZE_K 4 // 线程块的大小

__global__ void SgemmWithSharedmem(int *A, int *B, int *C, int M, int N,int K);