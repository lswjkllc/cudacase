
void SgemmWithCPU(int *A, int *B, int *C, int M, int N, int K) {
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            int sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
}
