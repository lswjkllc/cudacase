#ifndef CUDA_GEMM_SHAREDMEM_H
#define CUDA_GEMM_SHAREDMEM_H

# include <cuda_runtime.h>

template <const int BLOCK_SIZE_K>
__global__ void SgemmWithSharedmem(int *A, int *B, int *C, int M, int N,int K);

// 矩阵乘法分块
// 把数据搬到更快的存储器中（比如共享内存），共享内存的大小有限，利用分块实现对共享内存的利用
// grid : (M/BLOCK_SIZE_K,N/BLOCK_SIZE_K)   block : (BLOCK_SIZE_K,BLOCK_SIZE_K)
template <const int BLOCK_SIZE_K>
__global__ void SgemmWithSharedmem(int *A, int *B, int *C, int M, int N,int K) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;

    __shared__ int smem_a[BLOCK_SIZE_K][BLOCK_SIZE_K];
    __shared__ int smem_b[BLOCK_SIZE_K][BLOCK_SIZE_K];

    // 每个block负责C中一个维度为的小矩阵块的计算,计算中一共有k(K/BLOCK_SIZE_K)次迭代
    // 每一次迭代都需要读取A中一个维度为BLOCK_SIZE_K*BLOCK_SIZE_K的小矩阵块和B中一个维度为BLOCK_SIZE_K*BLOCK_SIZE_K的小矩阵块
    int sum = 0;
    for(int i = 0; i <= K / BLOCK_SIZE_K; i++){
        int ida = row * K + i * BLOCK_SIZE_K + threadIdx.y; // A数据的索引

        if (row < M && BLOCK_SIZE_K * i + threadIdx.y < K) {
            smem_a[threadIdx.x][threadIdx.y] = A[ida];
        } else {
            smem_a[threadIdx.x][threadIdx.y] = 0;
        }

        int idb = (threadIdx.x + i * BLOCK_SIZE_K) * N + col; // B数据的索引
        if (col < N && BLOCK_SIZE_K * i + threadIdx.x < K) {
            smem_b[threadIdx.x][threadIdx.y] = B[idb];
        } else {
            smem_b[threadIdx.x][threadIdx.y] = 0;
        }

        __syncthreads(); // 等待线程块的共享内存写入数据
#pragma unroll
        for (int i = 0; i < BLOCK_SIZE_K; i++) {
            sum += smem_a[threadIdx.x][i] * smem_b[i][threadIdx.y];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

#endif // CUDA_GEMM_SHAREDMEM_H
