#include <cuda_runtime.h>
#include <iostream>
#include <cublas_v2.h>
#include "cuda_utils.h"

// CUDA kernel for matrix multiplication with A and transpose(B)
__global__ void matMulCudaKernel(float* A, float* B, float* C, int aRows, int aCols, int bRows) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < aRows && col < bRows) {
        float value = 0;
        for (int k = 0; k < aCols; ++k) {
            // B is accessed in transposed manner
            value += A[row * aCols + k] * B[col * aCols + k];
        }
        C[row * bRows + col] = value;
    }
}

extern "C" void matMulCUDA(float* a, int aRows, int aCols, float* b, int bRows, int bCols, float* out) {
    float *d_A, *d_B, *d_C;
    size_t sizeA = aRows * aCols * sizeof(float);
    size_t sizeB = bRows * bCols * sizeof(float);
    size_t sizeC = aRows * bRows * sizeof(float);

    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    cudaMemcpy(d_A, a, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b, sizeB, cudaMemcpyHostToDevice);

    // Cuda Kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid((bRows + dimBlock.x - 1) / dimBlock.x, (aRows + dimBlock.y - 1) / dimBlock.y);
    matMulCudaKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, aRows, aCols, bRows);

    cudaMemcpy(out, d_C, sizeC, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Cublas for matrix multiplication with A and transpose(B)
extern "C" void matMulCublas(float* a, int aRows, int aCols, float* b, int bRows, int bCols, float* out) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float *d_A, *d_B, *d_C;
    size_t sizeA = aRows * aCols * sizeof(float);
    size_t sizeB = bRows * bCols * sizeof(float);
    size_t sizeC = aRows * bRows * sizeof(float);

    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    cudaMemcpy(d_A, a, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b, sizeB, cudaMemcpyHostToDevice);
    
    // Cublas Kernel
    float one = 1.0;
    float zero = 0.0;

    // We WTG C = A * B.T
    // Cublas stores in column order while C stores in row order
    // So Cublas interprets A and B as A.T and B.T
    // Therefore we input B.T * A -> interpreted as B * A.T = C.T
    // C.T in column major = C in row major, so we have what we want
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                bRows, aRows, aCols, // rows C, cols C, cols op(A)
                &one,
                d_B, bCols, // ld B
                d_A, aCols, // ld A
                &zero, d_C, bRows); // ld C

    cudaMemcpy(out, d_C, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
}

// Take a slice out of a larger matrix and return a new matrix with the given shape
extern "C" Matrix sliceCublas(Matrix a, int b, int rows, int cols) {
    // change to devicetodevice later TODO
    Matrix out = {a.dat + b * rows, rows, cols};
    return out;
}