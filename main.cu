# include <iostream>
# include <cmath>
# include <cuda_runtime.h>

# include "host/gemm.h"
# include "cuda/gemm/globalmem_2dmm.cuh"
# include "cuda/gemm/sharedmem_2dmm.cuh"

# define FLOAT_DIFF_EPSILON 1e-5

void printReult(float *C, int M, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
}

bool equalResult(const float *A, const float *B, int M, int N) {
    float a, b, diff;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            a = A[i * N + j];
            b = B[i * N + j];
            diff = fabs(a - b);
            if (diff >= FLOAT_DIFF_EPSILON) {
                return false;
            }
        }
    }
    return true;
}

int main() {
    // Initialize matrices A, B, and C
    constexpr int M = 512, N = 256, K = 128;

    float *A, *B, *C;
    // Allocate host memory
    A = (float *)malloc(M * K * sizeof(int));
    B = (float *)malloc(K * N * sizeof(int));
    C = (float *)malloc(M * N * sizeof(int));
    // Initialize matrices A and B with some values
    for (int i = 0; i < M * K; ++i) {
        A[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    for (int i = 0; i < K * N; ++i) {
        B[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    for (int i = 0; i < M * N; ++i) {
        C[i] = static_cast<float>(0);
    }
    // int A[M * K] = {1, 2, 3, 4, 5, 6, 7, 8};
    // int B[K * N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    // int C[M * N] = {0};

    // Call the matrix multiplication function
    SgemmWithCPU(A, B, C, M, N, K);
    // // Print the result
    // std::cout << "HOST Result: " << std::endl;
    // printReult(C, M, N);
    // std::cout << std::endl;

    // Call the CUDA matrix multiplication function
    float *d_A, *d_B, *d_C;
    // Allocate device memory
    cudaMalloc((void**)&d_A, M * K * sizeof(int));
    cudaMalloc((void**)&d_B, K * N * sizeof(int));
    cudaMalloc((void**)&d_C, M * N * sizeof(int));
    // Copy matrices A, B, and C from host to device
    cudaMemcpy(d_A, A, M * K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, M * N * sizeof(int), cudaMemcpyHostToDevice);
    
    // ------------------------ calculate grid and block size ---------------------------------------------
    constexpr int BLOCK_SIZE = 16;
    dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE); // celi value
    dim3 block(BLOCK_SIZE, BLOCK_SIZE); // 16 * 16 = 256 threads per block

    // ------------------------ global memory 2D matrix multiplication ---------------------------------------------
    // Define the result matrix
    float *Ret_1 = (float *)malloc(M * N * sizeof(int));
    for (int i = 0; i < M * N; ++i) {
        Ret_1[i] = 0;
    }
    // Call the CUDA matrix multiplication function
    SgemmWithGlobalmem<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    // // Synchronize the device
    // cudaDeviceSynchronize();
    // Copy the result back to the host 
    cudaMemcpy(Ret_1, d_C, M * N * sizeof(int), cudaMemcpyDeviceToHost);
    // Print the result
    std::cout << "Global memory 2dmm CUDA Result: ";
    equalResult(Ret_1, C, M, N) ? std::cout << "Equal!\n" : std::cout << "Not Equal!\n";
    // Free the result matrix
    free(Ret_1);

    // --------------------- shared memory 2D matrix multiplication ------------------------------------------------
    // Define the result matrix
    float *Ret_2 = (float *)malloc(M * N * sizeof(int));
    for (int i = 0; i < M * N; ++i) {
        Ret_2[i] = 0;
    }
    // Call the CUDA matrix multiplication function
    SgemmWithSharedmem<BLOCK_SIZE><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    // // Synchronize the device
    // cudaDeviceSynchronize();
    // Copy the result back to the host
    cudaMemcpy(Ret_2, d_C, M * N * sizeof(int), cudaMemcpyDeviceToHost);
    // Print the result
    std::cout << "Shared memory 2dmm CUDA Result: ";
    equalResult(Ret_2, C, M, N) ? std::cout << "Equal!\n" : std::cout << "Not Equal!\n";
    // Free the result matrix
    free(Ret_2);

    // --------------------------- free device/host memory ---------------------------------------------------------

    // Free device memory
    cudaFree(d_A);  
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(A);
    free(B);
    free(C);

    return EXIT_SUCCESS;
}
