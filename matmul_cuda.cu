#include <cuda_runtime.h>
#include "matmul_cuda.h"

// CUDA kernel for matrix multiplication
__global__ void matMulKernel(float* A, float* B, float* C, int aRows, int aCols, int bCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < aRows && col < bCols) {
        float value = 0;
        for (int k = 0; k < aCols; ++k) {
            value += A[row * aCols + k] * B[k * bCols + col];
        }
        C[row * bCols + col] = value;
    }
}

extern "C" void matMulCUDA(float* a, int aRows, int aCols, float* b, int bRows, int bCols, float* out) {
    float *d_A, *d_B, *d_C;
    size_t sizeA = aRows * aCols * sizeof(float);
    size_t sizeB = bRows * bCols * sizeof(float);
    size_t sizeC = aRows * bCols * sizeof(float);

    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    cudaMemcpy(d_A, a, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b, sizeB, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((bCols + dimBlock.x - 1) / dimBlock.x, (aRows + dimBlock.y - 1) / dimBlock.y);

    matMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, aRows, aCols, bCols);

    cudaMemcpy(out, d_C, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
