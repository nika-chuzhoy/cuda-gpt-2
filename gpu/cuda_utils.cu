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
    dim3 dimBlock(32, 32);
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

// From Lab 2
__global__
void transposeKernel(const float *input, float *output, int rows, int cols) {
    const int TILE_DIM = 64;
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 for padding to avoid bank conflicts

    // Global index calculations for reading input
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + 4 * threadIdx.y;  // Each thread reads 4 elements along y

    // Local index within shared memory
    int localX = threadIdx.x;
    int localY = threadIdx.y;

    // Read input matrix in a coalesced manner and store into shared memory
    if (xIndex < cols) {
        if (yIndex + 0 < rows) tile[localY * 4 + 0][localX] = input[xIndex + (yIndex + 0) * cols];
        if (yIndex + 1 < rows) tile[localY * 4 + 1][localX] = input[xIndex + (yIndex + 1) * cols];
        if (yIndex + 2 < rows) tile[localY * 4 + 2][localX] = input[xIndex + (yIndex + 2) * cols];
        if (yIndex + 3 < rows) tile[localY * 4 + 3][localX] = input[xIndex + (yIndex + 3) * cols];
    }

    __syncthreads();  // Synchronize to ensure all writes to shared memory are complete

    // Transpose within shared memory
    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + 4 * threadIdx.y;

    // Write output in a coalesced manner
    if (xIndex < rows) {
        if (yIndex + 0 < cols) output[(yIndex + 0) * rows + xIndex] = tile[localX][localY * 4 + 0];
        if (yIndex + 1 < cols) output[(yIndex + 1) * rows + xIndex] = tile[localX][localY * 4 + 1];
        if (yIndex + 2 < cols) output[(yIndex + 2) * rows + xIndex] = tile[localX][localY * 4 + 2];
        if (yIndex + 3 < cols) output[(yIndex + 3) * rows + xIndex] = tile[localX][localY * 4 + 3];
    }
}

extern "C" void transposeCUDA(Matrix a, Matrix out)
{
    float *d_input, *d_output;
    size_t size = a.rows * a.cols * sizeof(float);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, a.dat, size, cudaMemcpyHostToDevice);

    const int TILE_DIM = 64;

    dim3 dimBlock(TILE_DIM, TILE_DIM / 4); // 64x16
    dim3 dimGrid((a.cols + TILE_DIM - 1) / TILE_DIM, (a.rows + TILE_DIM - 1) / TILE_DIM);

    transposeKernel<<<dimGrid, dimBlock>>>(d_input, d_output, a.rows, a.cols);

    cudaMemcpy(out.dat, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

//  Matrix fn(Matrix a, float k)
#define UNARY(fn, opr)                                                 \
    __global__ void fn##Kernel(float* a, int aRows, int aCols, float* out, float k) { \
        int row = blockIdx.y * blockDim.y + threadIdx.y;               \
        int col = blockIdx.x * blockDim.x + threadIdx.x;               \
        if (row < aRows && col < aCols) {                              \
            int i = row * aCols + col;                                 \
            float b = a[i];                                            \
            out[i] = opr;                                              \
        }                                                              \
    }                                                                  \
    extern "C" Matrix fn##CUDA(Matrix m, float k) {                    \
        float* a = m.dat;                                              \
        int aRows = m.rows;                                            \
        int aCols = m.cols;                                            \
        float *d_a;                                                    \
        size_t sizeA = aRows * aCols * sizeof(float);                  \
        cudaMalloc((void**)&d_a, sizeA);                               \
        cudaMemcpy(d_a, a, sizeA, cudaMemcpyHostToDevice);             \
        dim3 blockSize(16, 16);                                        \
        dim3 gridSize((aCols + blockSize.x - 1) / blockSize.x,         \
                      (aRows + blockSize.y - 1) / blockSize.y);        \
        fn##Kernel<<<gridSize, blockSize>>>(d_a, aRows, aCols, d_a, k);\
        cudaMemcpy(m.dat, d_a, sizeA, cudaMemcpyDeviceToHost);         \
        cudaFree(d_a);                                                 \
        return m;                                                      \
    }

UNARY(divide_const, b / k)                      // divide by a constant
UNARY(add_const, b + k)                         // add a constant
UNARY(mat_isqrt, 1. / sqrt(b))                  // square root each entry
UNARY(mat_exp, exp(b))                          // exponetiate each entry
UNARY(broadcast, a[(i / aCols) * aCols])  // copy the first column to every column

// Tril is the first of two special functions.
//   a   b   c        exp(a/8) exp(b/8) exp(c/8)
//   d   e   f   ->      0     exp(e/8) exp(f/8)
//   g   h   i           0        0        0
// it's use will be described later
UNARY(tril, (i / k < i % (int)k) ? 0 : exp(b / 8))

// GELU is the activation function used for transformers
UNARY(GELU, b / 2 * (1 + tanh(.7978845 * (b + .044715 * b * b * b))))