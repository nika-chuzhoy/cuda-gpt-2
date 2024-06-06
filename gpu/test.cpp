#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstdio> 
#include "cuda_utils.h"


bool compareMatrices(float *a, float *b, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        if (fabs(a[i] - b[i]) > 1e-5) {
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

void cpuMatMul(float* a, int aRows, int aCols, float* b, int bRows, int bCols, float* out) {
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
    const int aRows = 5;
    const int aCols = 3;

    const int bRows = 4;
    const int bCols = 3;

    float *a_input = generateRandomMatrix(aRows, aCols);
    float *b_input = generateRandomMatrix(bRows, bCols);

    float *c_output_gpu = (float*) malloc(aRows * bRows * sizeof(float));
    float *c_output_cpu = (float*) malloc(aRows * bRows * sizeof(float));

    matMulCUDA(a_input, aRows, aCols, b_input, bRows, bCols, c_output_gpu);
    cpuMatMul(a_input, aRows, aCols, b_input, bRows, bCols, c_output_cpu);

    // printMatrix(c_output_gpu, aRows, bRows);
    // printMatrix(c_output_cpu, aRows, bRows);

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
    const int aRows = 5;
    const int aCols = 3;

    const int bRows = 2;
    const int bCols = 3;

    float *a_input = generateRandomMatrix(aRows, aCols);
    float *b_input = generateRandomMatrix(bRows, bCols);

    float *c_output_gpu = new float[aRows * bRows];
    float *c_output_cpu = new float[aRows * bRows];

    matMulCublas(a_input, aRows, aCols, b_input, bRows, bCols, c_output_gpu);
    cpuMatMul(a_input, aRows, aCols, b_input, bRows, bCols, c_output_cpu);

    // printMatrix(c_output_gpu, aRows, bRows);
    // printMatrix(c_output_cpu, aRows, bRows);

    if (compareMatrices(c_output_gpu, c_output_cpu, aRows, bRows)) {
        std::cout << "Test Cublas matmul PASSED." << std::endl;
    } else {
        std::cout << "Test Cublas matmul FAILED." << std::endl;
    }

    delete[] a_input;
    delete[] b_input;
    delete[] c_output_gpu;
    delete[] c_output_cpu;
}

void cpuTranspose(float *input, float *output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

void cudaTransposeTest() {
    const int rows = 256;
    const int cols = 256;

    float *h_input = generateRandomMatrix(rows, cols);
    float *h_output_gpu = new float[rows * cols];
    float *h_output_cpu = new float[rows * cols];

    Matrix mat;
    mat.dat = h_input;
    mat.rows = rows;
    mat.cols = cols;

    cudaTranspose(mat);
    cpuTranspose(h_input, h_output_cpu, rows, cols);
    

    if (compareMatrices(mat.dat, h_output_cpu, rows, cols)) {
        std::cout << "Test PASSED." << std::endl;
    } else {
        std::cout << "Test FAILED." << std::endl;
    }

    delete[] h_input;
    delete[] h_output_gpu;
    delete[] h_output_cpu;
}

int main() {

    matMulCUDATest();
    matMulCublasTest();
    // cudaTransposeTest();

    return 0;
}