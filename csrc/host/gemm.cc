
void matrix_multiply_cpu(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int h = 0; h < K; ++h) {
                sum += A[i * K + h] * B[h * N + j];
            }
            C[i * N + j] = sum;
        }
}
