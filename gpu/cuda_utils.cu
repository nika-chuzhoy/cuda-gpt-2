#include <cuda_runtime.h>
#include <iostream>

// CUDA kernel for matrix multiplication with A and transpose(B)
__global__ void matMulKernel(float* A, float* B, float* C, int aRows, int aCols, int bRows) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < aRows && col < bRows) { // Notice the change from bCols to bRows
        float value = 0;
        for (int k = 0; k < aCols; ++k) {
            // B is accessed in transposed manner
            value += A[row * aCols + k] * B[col * aCols + k]; // Accessing B[col][k] instead of B[k][col]
        }
        C[row * bRows + col] = value; // bCols changed to bRows
    }
}

extern "C" void matMulCUDA(float* a, int aRows, int aCols, float* b, int bRows, int bCols, float* out) {
    if (aCols != bCols) { // Adjust the condition to check aCols against bCols, since B is transposed
        std::cerr << "Matrix dimensions mismatch: aCols must equal bCols for transpose multiplication." << std::endl;
        return;
    }

    float *d_A, *d_B, *d_C;
    size_t sizeA = aRows * aCols * sizeof(float);
    size_t sizeB = bRows * bCols * sizeof(float);
    size_t sizeC = aRows * bRows * sizeof(float); // bCols changed to bRows

    cudaError_t status;
    status = cudaMalloc((void**)&d_A, sizeA);
    if (status != cudaSuccess) { std::cerr << "CUDA malloc failed for d_A" << std::endl; return; }
    status = cudaMalloc((void**)&d_B, sizeB);
    if (status != cudaSuccess) { std::cerr << "CUDA malloc failed for d_B" << std::endl; cudaFree(d_A); return; }
    status = cudaMalloc((void**)&d_C, sizeC);
    if (status != cudaSuccess) { std::cerr << "CUDA malloc failed for d_C" << std::endl; cudaFree(d_A); cudaFree(d_B); return; }

    cudaMemcpy(d_A, a, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b, sizeB, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((bRows + dimBlock.x - 1) / dimBlock.x, (aRows + dimBlock.y - 1) / dimBlock.y);

    matMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, aRows, aCols, bRows); // Changed bCols to bRows
    status = cudaGetLastError();
    if (status != cudaSuccess) { std::cerr << "Kernel launch failed: " << cudaGetErrorString(status) << std::endl; cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); return; }

    cudaMemcpy(out, d_C, sizeC, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}