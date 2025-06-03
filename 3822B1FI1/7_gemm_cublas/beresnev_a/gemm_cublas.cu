#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    size_t bytes = n * n * sizeof(float);
    std::vector<float> c(n * n, 0.0f);
    float *d_a, *d_b, *d_c, *d_ct;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMalloc(&d_ct, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, 
                n, n, n, 
                &alpha, 
                d_a, n, 
                d_b, n, 
                &beta, 
                d_c, n);
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                n, n, &alpha, 
                d_c, n, &beta, nullptr, 
                n, d_ct, n);

    cudaMemcpy(c.data(), d_ct, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_ct);
    cublasDestroy(handle);

    return c;
}
