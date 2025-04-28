# include <iostream>
# include <cuda_runtime.h>

# include "host/gemm.h"
# include "device/gemm/native_2dmm.cuh"

int main() {
    // Initialize matrices A, B, and C
    const int M = 2, N = 3, K = 4;
    float A[M * K] = {1, 2, 3, 4, 5, 6, 7, 8};
    float B[K * N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    float C[M * N] = {0};

    // Call the matrix multiplication function
    matrix_multiply_cpu(A, B, C, M, N, K);
    // Print the result
    std::cout << "HOST Result:" << std::endl;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Call the CUDA matrix multiplication function
    float *d_A, *d_B, *d_C;
    // Allocate device memory
    cudaMalloc((void**)&d_A, M * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * N * sizeof(float));
    cudaMalloc((void**)&d_C, M * N * sizeof(float));
    // Copy matrices A, B, and C from host to device
    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, M * N * sizeof(float), cudaMemcpyHostToDevice);
    // Call the CUDA matrix multiplication function
    dim3 blocksPerGrid(1);
    dim3 threadsPerBlock(M * N);
    matrix_multiply_cuda_naive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    // // Synchronize the device
    // cudaDeviceSynchronize();
    // Copy the result back to the host 
    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    // Print the result
    std::cout << "CUDA Result:" << std::endl;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
    // Free device memory
    cudaFree(d_A);  
    cudaFree(d_B);
    cudaFree(d_C);

    return EXIT_SUCCESS;
}
