# include <cuda_runtime.h>

// 矩阵乘法的逐点实现方式
// 对于矩阵A（m * k）和矩阵B（k * n），每个元素访问的次数分别是n与m。这里存在着对全局内存的多次访问
// 2D
__global__ void SgemmWithGlobalmem(float *A, float *B, float *C, int M, int N,int K) {
    // // Impl Method One: ----------------------------------------------------------------------
    // // Calculate the row and column indices for the output matrix C
    // int row = threadIdx.x + blockIdx.x * blockDim.x;
    // int col = threadIdx.y + blockIdx.y * blockDim.y;
    // // Check whether the indices are within the bounds of the output matrix C
    // if (row >= M || col >= N)
    //     return;

    // // Calculate the element value of C
    // int sum = 0;
    // for (int k = 0; k < K; k++) {
    //     sum += A[row * K + k] * B[k * N + col];
    // }
    // // Store the result in the output matrix C: to HBM
    // C[row * N + col] = sum;

    // Impl Method Two: ----------------------------------------------------------------------
    // Calculate the row and column indices for the output matrix C
    const int row = threadIdx.x + blockDim.x * blockIdx.x;
    const int col = threadIdx.y + blockDim.y * blockIdx.y;
    // Check whether the indices are within the bounds of the output matrix C
    if (row >= M || col >= N)
        return;
    // Caluculate the start address in A and B for the current block
    float *A_start = A + blockDim.x * blockIdx.x * K;
    float *B_start = B + blockDim.y * blockIdx.y;
    // Calculate the element value of C
    int sum = 0;
    for (int k = 0; k < K; k++) {
        sum += A_start[threadIdx.x * K + k] * B_start[k * N + threadIdx.y];
    }
    // Store the result in the output matrix C: to HBM
    C[row * N + col] = sum;
}
