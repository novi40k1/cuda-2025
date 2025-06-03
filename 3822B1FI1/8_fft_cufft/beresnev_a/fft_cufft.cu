#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

__global__ void normalizeKernel(float* array, int totalSize, float normalizationFactor) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < totalSize) {
        array[index] *= normalizationFactor;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int totalSize = input.size();
    int signalLength = totalSize / (2 * batch);
    std::vector<float> outputData(totalSize);
    int dataSize = sizeof(cufftComplex) * signalLength * batch;

    cufftHandle fftPlan;
    cufftComplex* deviceData;

    cudaMalloc(&deviceData, dataSize);
    cudaMemcpy(deviceData, input.data(), dataSize, cudaMemcpyHostToDevice);

    cufftPlan1d(&fftPlan, signalLength, CUFFT_C2C, batch);
    cufftExecC2C(fftPlan, deviceData, deviceData, CUFFT_FORWARD);
    cufftExecC2C(fftPlan, deviceData, deviceData, CUFFT_INVERSE);

    cudaDeviceProp deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, 0);
    size_t block = deviceProperties.maxThreadsPerBlock;
    size_t grid = (totalSize + block - 1) / block;

    float normalizationFactor = 1.0f / signalLength;
    normalizeKernel<<<grid, block>>>(reinterpret_cast<float*>(deviceData), totalSize, normalizationFactor);

    cudaMemcpy(outputData.data(), deviceData, dataSize, cudaMemcpyDeviceToHost);
    cufftDestroy(fftPlan);
    cudaFree(deviceData);

    return outputData;
}
