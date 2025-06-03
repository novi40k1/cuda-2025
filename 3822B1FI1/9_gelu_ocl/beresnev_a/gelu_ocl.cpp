#include "gelu_ocl.h"
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>

const char* geluKernelSource = R"(
__kernel void gelu(__global const float* input, __global float* output, int size) {
    int id = get_global_id(0);
    if (id < size) {
        float x = input[id];
        output[id] = 0.5f * x * (1.0f + tanh(sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
    }
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input) {
    size_t inputSize = input.size();
    std::vector<float> output(inputSize);

    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_mem inputBuffer = nullptr;
    cl_mem outputBuffer = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;

    cl_int err;

    err = clGetPlatformIDs(1, &platform, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get platform ID." << std::endl;
        return output;
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get device ID." << std::endl;
        return output;
    }

    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create context." << std::endl;
        return output;
    }

    queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create command queue." << std::endl;
        clReleaseContext(context);
        return output;
    }

    inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        inputSize * sizeof(float), (void*)input.data(), &err);
    outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        inputSize * sizeof(float), nullptr, &err);

    program = clCreateProgramWithSource(context, 1, &geluKernelSource, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create program." << std::endl;
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return output;
    }

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to build program." << std::endl;
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return output;
    }

    kernel = clCreateKernel(program, "gelu", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create kernel." << std::endl;
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return output;
    }

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &inputSize);

    size_t globalWorkSize = inputSize;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to enqueue kernel." << std::endl;
    }

    clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0,
        inputSize * sizeof(float), output.data(), 0, nullptr, nullptr);

    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return output;
}
