# include <iostream>
# include <cuda_runtime.h>

# include "host/gemm.h"
# include "device/gemm/native_2dmm.cuh"
# include "device/gemm/sharedmem_2dmm.cuh"

void printReult(int *C, int M, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
}

bool equalResult(int *A, int *B, int M, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (A[i * N + j] != B[i * N + j]) {
                return false;
            }
        }
    }
    return true;
}

int main() {
    // Initialize matrices A, B, and C
    const int M = 32, N = 16, K = 8;

    int *A, *B, *C;
    // Allocate host memory
    A = (int*)malloc(M * K * sizeof(int));
    B = (int*)malloc(K * N * sizeof(int));
    C = (int*)malloc(M * N * sizeof(int));
    // Initialize matrices A and B with some values
    for (int i = 0; i < M * K; ++i) {
        A[i] = 1;
    }
    for (int i = 0; i < K * N; ++i) {
        B[i] = 1;
    }
    for (int i = 0; i < M * N; ++i) {
        C[i] = 0.0f;
    }
    // int A[M * K] = {1, 2, 3, 4, 5, 6, 7, 8};
    // int B[K * N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    // int C[M * N] = {0};

    // Call the matrix multiplication function
    matrix_multiply_cpu(A, B, C, M, N, K);
    // Print the result
    std::cout << "HOST Result: " << std::endl;
    printReult(C, M, N);
    std::cout << std::endl;

    // Call the CUDA matrix multiplication function
    int *d_A, *d_B, *d_C;
    // Allocate device memory
    cudaMalloc((void**)&d_A, M * K * sizeof(int));
    cudaMalloc((void**)&d_B, K * N * sizeof(int));
    cudaMalloc((void**)&d_C, M * N * sizeof(int));
    // Copy matrices A, B, and C from host to device
    cudaMemcpy(d_A, A, M * K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, M * N * sizeof(int), cudaMemcpyHostToDevice);

    // ------------------------ naive 2D matrix multiplication ------------------------
    // Define the result matrix
    int *Ret_1 = (int*)malloc(M * N * sizeof(int));
    for (int i = 0; i < M * N; ++i) {
        Ret_1[i] = 0.0f;
    }
    // Call the CUDA matrix multiplication function
    dim3 blocksPerGrid_1(1);
    dim3 threadsPerBlock_1(M * N);
    SgemmWithNative<<<blocksPerGrid_1, threadsPerBlock_1>>>(d_A, d_B, d_C, M, N, K);
    // // Synchronize the device
    // cudaDeviceSynchronize();
    // Copy the result back to the host 
    cudaMemcpy(Ret_1, d_C, M * N * sizeof(int), cudaMemcpyDeviceToHost);
    // Print the result
    std::cout << "Native    2dmm CUDA Result: ";
    equalResult(Ret_1, C, M, N) ? std::cout << "Equal!\n" : std::cout << "Not Equal!\n";
    // Free the result matrix
    free(Ret_1);

    // --------------------- shared memory 2D matrix multiplication ------------------------
    // Define the result matrix
    int *Ret_2 = (int*)malloc(M * N * sizeof(int));
    for (int i = 0; i < M * N; ++i) {
        Ret_2[i] = 0.0f;
    }
    // Call the CUDA matrix multiplication function
    const int BLOCK_SIZE_K = 8;
    dim3 blocksPerGrid_2(M / BLOCK_SIZE_K, N / BLOCK_SIZE_K);
    dim3 threadsPerBlock_2(BLOCK_SIZE_K, BLOCK_SIZE_K);
    SgemmWithSharedmem<BLOCK_SIZE_K><<<blocksPerGrid_2, threadsPerBlock_2>>>(d_A, d_B, d_C, M, N, K);
    // // Synchronize the device
    // cudaDeviceSynchronize();
    // Copy the result back to the host
    cudaMemcpy(Ret_2, d_C, M * N * sizeof(int), cudaMemcpyDeviceToHost);
    // Print the result
    std::cout << "Sharedmem 2dmm CUDA Result: ";
    equalResult(Ret_2, C, M, N) ? std::cout << "Equal!\n" : std::cout << "Not Equal!\n";
    // Free the result matrix
    free(Ret_2);

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
