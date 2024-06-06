#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstdio> 
#include "cuda_utils.h"
#include <time.h>
#include <cuda_runtime.h>

using namespace std;

bool compareMatrices(float *a, float *b, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        if (fabs(a[i] - b[i]) > 1e-2) {
            return false;
        }
    }
    return true;
}

float* generateRandomMatrix(int rows, int cols) {
    srand(42);
    float* matrix = (float*)malloc(rows * cols * sizeof(float));
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = 10 * static_cast<float>(rand()) / RAND_MAX;
    }
    return matrix;
}

void matMulCPU(float* a, int aRows, int aCols, float* b, int bRows, int bCols, float* out) {
   for (int i = 0; i < aRows; i++) {
        for (int j = 0; j < bRows; j++) {
            float sum = 0;
            for (int k = 0; k < aCols; k++) {
                sum += a[i * aCols + k] * b[j * bCols + k];
                if (j == 3) {
              }
            }
            out[i * bRows + j] = sum;
        }
    }
}

void printMatrix(float* m, int mRows, int mCols) {
    printf("---\n");
    for (int i = 0; i < mRows; ++i) {
        for (int j = 0; j < mCols; ++j) {
            printf("%f ", m[i * mCols + j]);
        }
        printf("\n");
    }
    printf("---\n");
}

void matMulCUDATest() {
    std::cout << "------------------------------------------" << std::endl;
    std::cout << "Test CUDA matmul RUNNING." << std::endl;
    const int aRows = 500;
    const int aCols = 300;

    const int bRows = 400;
    const int bCols = 300;

    float *a_input = generateRandomMatrix(aRows, aCols);
    float *b_input = generateRandomMatrix(bRows, bCols);

    float *c_output_gpu = (float*) malloc(aRows * bRows * sizeof(float));
    float *c_output_cpu = (float*) malloc(aRows * bRows * sizeof(float));
    
    // Use the CUDA machinery for recording time
    cudaEvent_t start_cpu, stop_cpu;
    cudaEventCreate(&start_cpu);
    cudaEventCreate(&stop_cpu);
    cudaEventRecord(start_cpu);

    cpuMatMul(a_input, aRows, aCols, b_input, bRows, bCols, c_output_cpu);

    cudaEventRecord(stop_cpu);
    cudaEventSynchronize(stop_cpu);
    float cpu_time_milliseconds;
    cudaEventElapsedTime(&cpu_time_milliseconds, start_cpu, stop_cpu);

    float gpu_time_milliseconds = matMulCUDA(a_input, aRows, aCols, b_input, bRows, bCols, c_output_gpu);
    // printMatrix(c_output_gpu, aRows, bRows);
    // printMatrix(c_output_cpu, aRows, bRows);

    cout << endl;
    cout << "CPU time: " << cpu_time_milliseconds << " milliseconds" << endl;
    cout << "GPU time: " << gpu_time_milliseconds << " milliseconds" << endl;
    cout << endl << "Speedup factor: " <<
        cpu_time_milliseconds / gpu_time_milliseconds << endl << endl;

    if (compareMatrices(c_output_gpu, c_output_cpu, aRows, bRows)) {
        std::cout << "Test CUDA matmul PASSED." << std::endl;
    } else {
        std::cout << "Test CUDA matmul FAILED." << std::endl;
    }

    free(a_input);
    free(b_input);
    free(c_output_gpu);
    free(c_output_cpu);
}

void matMulCublasTest() {
    std::cout << "------------------------------------------" << std::endl;
    std::cout << "Test Cublas matmul RUNNING." << std::endl;

    const int aRows = 500;
    const int aCols = 300;

    const int bRows = 400;
    const int bCols = 300;

    float *a_input = generateRandomMatrix(aRows, aCols);
    float *b_input = generateRandomMatrix(bRows, bCols);

    float *c_output_gpu = (float*) malloc(aRows * bRows * sizeof(float));
    float *c_output_cpu = (float*) malloc(aRows * bRows * sizeof(float));

    float cuda_time = matMulCUDA(a_input, aRows, aCols, b_input, bRows, bCols, c_output_gpu);
    float cublas_time = matMulCublas(a_input, aRows, aCols, b_input, bRows, bCols, c_output_cpu);

    cout << endl;
    cout << "CUBLAS time: " << cublas_time << " milliseconds" << endl;
    cout << "CUDA time: " << cuda_time << " milliseconds" << endl;
     cout << endl << "Speedup factor: " <<
        cublas_time / cuda_time << endl << endl;

    // printMatrix(c_output_gpu, aRows, bRows);
    // printMatrix(c_output_cpu, aRows, bRows);

    if (compareMatrices(c_output_gpu, c_output_cpu, aRows, bRows)) {
        std::cout << "Test Cublas matmul PASSED." << std::endl;
    } else {
        std::cout << "Test Cublas matmul FAILED." << std::endl;
    }

    free(a_input);
    free(b_input);
    free(c_output_gpu);
    free(c_output_cpu);
}

void transposeCPU(float *input, float *output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}


void cudaTransposeTest() {
    const int rows = 77;
    const int cols = 80;

    float *h_input = generateRandomMatrix(rows, cols);
    float *h_output_cpu = (float*) malloc(rows * cols  * sizeof(float));

    Matrix mat;
    mat.dat = h_input;
    mat.rows = rows;
    mat.cols = cols;

    transposeCPU(h_input, h_output_cpu, rows, cols);
    transposeCUDA(mat);

    // printMatrix(h_output_cpu, cols, rows);
    // printMatrix(mat.dat, cols, rows);

    if (compareMatrices(mat.dat, h_output_cpu, cols, rows)) {
        std::cout << "Test Transpose PASSED." << std::endl;
    } else {
        std::cout << "Test Transpose FAILED." << std::endl;
    }

    free(h_input);
    free(h_output_cpu);
}

int main() {

    matMulCUDATest();
    matMulCublasTest();
    cudaTransposeTest();

    return 0;
}