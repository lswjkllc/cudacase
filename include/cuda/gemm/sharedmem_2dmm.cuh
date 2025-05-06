#ifndef CUDA_GEMM_SHAREDMEM_H
#define CUDA_GEMM_SHAREDMEM_H

# include <cuda_runtime.h>

template <const int BLOCK_SIZE>
__global__ void SgemmWithSharedmemV1(float *A, float *B, float *C, int M, int N,int K);

template <const int BLOCK_SIZE, const int STRIDE>
__global__ void SgemmWithSharedmemV2(float *A, float *B, float *C, int M, int N,int K);

// 矩阵乘法分块
// 把数据搬到更快的存储器中（比如共享内存），共享内存的大小有限，利用分块实现对共享内存的利用
// grid : (M/BLOCK_SIZE,N/BLOCK_SIZE)
// block: (BLOCK_SIZE, BLOCK_SIZE)
template <const int BLOCK_SIZE>
__global__ void SgemmWithSharedmemV1(float *A, float *B, float *C, int M, int N,int K) {
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

// Based on SgemmWithSharedmemV1, let each thread process more data, the increased stride is: STRIDE
// Each thread's data: STRIDE * STRIDE
// Each thread's step: BLOCK_SIZE * STRIDE
template <const int BLOCK_SIZE, const int STRIDE>
__global__ void SgemmWithSharedmemV2(float *A, float *B, float *C, int M, int N,int K) {
    // Rename the thread index for better readability
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // Calculate the row and column indices for the output matrix C
    int row = tx + blockIdx.x * blockDim.x;
    int col = ty + blockIdx.y * blockDim.y;

    // // Calculate the start address in A and B for the current block
    // float *A_start = A + blockDim.x * blockIdx.x * K;
    // float *B_start = B + blockDim.y * blockIdx.y;

    const int STEP = BLOCK_SIZE * STRIDE;
    __shared__ int smem_a[STEP][STEP];
    __shared__ int smem_b[STEP][STEP];

    float sum[STRIDE][STRIDE] = {0.0f};
    for(int step_offset = 0; step_offset < K; step_offset += STEP) {
        for (int i = 0; i < STRIDE; i++) {
            for (int j = 0; j < STRIDE; j++) {
                // Calculate the row and column indices for the shared memory
                int smemRow = tx + i * BLOCK_SIZE;
                int smemCol = ty + j * BLOCK_SIZE;

                // Calcuate the row and column indices for the input matrix A
                int aRow = row + i * STEP;
                int aCol = step_offset + j * STEP;
                if (aRow < M && aCol < K) {
                    // Calculate the index for the input matrix A
                    int ida = aRow * K + aCol;
                    // Load the data from global memory to shared memory
                    smem_a[smemRow][smemCol] = A[ida];
                }

                // Calculate the row and column indices for the input matrix B
                int bRow = step_offset + i * STEP;
                int bCol = col + j * STEP;
                if (bRow < K && bCol < N) {
                    // Calculate the index for the input matrix B
                    int idb = bRow * N + bCol;
                    // Load the data from global memory to shared memory
                    smem_b[smemRow][smemCol] = B[idb];
                }
            }
        }

        __syncthreads();

#pragma unroll
        for (int i = 0; i < STRIDE; i++) {
            for (int j = 0; j < STRIDE; j++) {
                for (int k = 0; k < BLOCK_SIZE; k++) {
                    // Calculate the row and column indices for the 'smea_a' shared memory
                    int semaARow = tx + i * BLOCK_SIZE;
                    int semaACol = k + j * BLOCK_SIZE;
                    //  Calculate the row and column indices for the 'smem_b' shared memory
                    int semaBRow = k + i * BLOCK_SIZE;
                    int semaBCol = ty + j * BLOCK_SIZE;

                    sum[i][j] += smem_a[semaARow][semaACol] * smem_b[semaBRow][semaBCol];
                }
            }
        }

        __syncthreads();
    }

    // Store the result in the output matrix C: to HBM
    for (int i = 0; i < STRIDE; i++) {
        for (int j = 0; j < STRIDE; j++) {
            int cRow = row + i * STEP;
            int cCol = col + j * STEP;

            if (cRow < M && cCol < N) {
                int idc = cRow * N + cCol;
                C[idc] = sum[i][j];
            }
        }
    }
}

#endif // CUDA_GEMM_SHAREDMEM_H
