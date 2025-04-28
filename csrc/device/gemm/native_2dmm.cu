# include <cuda_runtime.h>

// 矩阵乘法的逐点实现方式
// 对于矩阵A（m * k）和矩阵B（k * n），每个元素访问的次数分别是n与m。这里存在着对全局内存的多次访问
// 2D
__global__ void SgemmWithNative(int *a, int *b, int *c, int M, int N,int K) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;

    if (row >= M || col >= N)
        return;

    int value = 0.0;
    for (int i = 0; i < K; i++) {
        value += a[row * K + i] * b[i * N + col];
    }
    c[row * N + col] = value;
}
