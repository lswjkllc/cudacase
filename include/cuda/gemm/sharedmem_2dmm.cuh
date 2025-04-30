#ifndef CUDA_GEMM_SHAREDMEM_H
#define CUDA_GEMM_SHAREDMEM_H

# include <cuda_runtime.h>

template <const int BLOCK_SIZE>
__global__ void SgemmWithSharedmem(float *A, float *B, float *C, int M, int N,int K);

// 矩阵乘法分块
// 把数据搬到更快的存储器中（比如共享内存），共享内存的大小有限，利用分块实现对共享内存的利用
// grid : (M/BLOCK_SIZE,N/BLOCK_SIZE)
// block: (BLOCK_SIZE, BLOCK_SIZE)
template <const int BLOCK_SIZE>
__global__ void SgemmWithSharedmem(float *A, float *B, float *C, int M, int N,int K) {
    // Calculate the row and column indices for the output matrix C
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    if (row >= M || col >= N)
        return;

    // // Calculate the start address in A and B for the current block
    // float *A_start = A + blockDim.x * blockIdx.x * K;
    // float *B_start = B + blockDim.y * blockIdx.y;

    // Define the shared memory for A and B;
    // Each block share the shared memory for it's threads;
    // Shared memory size is BLOCK_SIZE * BLOCK_SIZE;
    __shared__ int smem_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int smem_b[BLOCK_SIZE][BLOCK_SIZE];

    // Each block is responsible for a small matrix block of size BLOCK_SIZE * BLOCK_SIZE in C
    // Each block's thread has K/BLOCK_SIZE iterations
    // Each block's thread's iteration need to read a small matrix block of size BLOCK_SIZE * BLOCK_SIZE in A and B
    // A and B matrix are flatten array
    int sum = 0;
    for(int block_offset = 0; block_offset < K; block_offset += BLOCK_SIZE) {
    // for(int k = 0; k <= K / BLOCK_SIZE; k++){
        // int block_offset = k * BLOCK_SIZE;
        int ida = row * K + block_offset + threadIdx.y; // the index of A matrix
        if (row < M && block_offset + threadIdx.y < K) {
            smem_a[threadIdx.x][threadIdx.y] = A[ida];
        } else {
            smem_a[threadIdx.x][threadIdx.y] = 0;
        }

        // block_offset = k * BLOCK_SIZE;
        int idb = (threadIdx.x + block_offset) * N + col; // the index of B matrix
        if (col < N && block_offset + threadIdx.x < K) {
            smem_b[threadIdx.x][threadIdx.y] = B[idb];
        } else {
            smem_b[threadIdx.x][threadIdx.y] = 0;
        }

        __syncthreads(); // Waiting for all threads in the block to finish writing to shared memory

#pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i++) {
            sum += smem_a[threadIdx.x][i] * smem_b[i][threadIdx.y];
        }

        __syncthreads(); // Waiting for all threads in the block to finish reading from shared memory
    }

    // Store the result in the output matrix C: to HBM
    C[row * N + col] = sum;
}

#endif // CUDA_GEMM_SHAREDMEM_H
