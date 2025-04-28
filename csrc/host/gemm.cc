
void matrix_multiply_cpu(int *A, int *B, int *C, int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            int sum = 0;
            for (int h = 0; h < K; ++h) {
                sum += A[i * K + h] * B[h * N + j];
            }
            C[i * N + j] = sum;
        }
}
