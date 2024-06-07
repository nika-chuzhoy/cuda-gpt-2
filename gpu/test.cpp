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
//    for (int i = 0; i < aRows; i++) {
//         for (int j = 0; j < bRows; j++) {
//             float sum = 0;
//             for (int k = 0; k < aCols; k++) {
//                 sum += a[i * aCols + k] * b[j * bCols + k];
//                 if (j == 3) {
//               }
//             }
//             out[i * bRows + j] = sum;
//         }
//     }

#ifdef GOFAST
#pragma omp parallel
#endif
    {
        for (int i = 0; i < aRows; i++) {
#ifdef GOFAST
#pragma omp for
#endif
            for (int j = 0; j < bRows; j += 4) {
                for (int k = 0; k < aCols; k += 4) {
                    for (int k2 = 0; k2 < 4; k2++)
                        for (int j2 = 0; j2 < 4; j2++)
                            out[i * bRows + j + j2] += a[i * aCols + k + k2] * b[(j + j2) * bCols + k + k2];
                }
            }
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

    matMulCPU(a_input, aRows, aCols, b_input, bRows, bCols, c_output_cpu);

    cudaEventRecord(stop_cpu);
    cudaEventSynchronize(stop_cpu);
    float cpu_time_milliseconds;
    cudaEventElapsedTime(&cpu_time_milliseconds, start_cpu, stop_cpu);

    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu);

    matMulCUDA(a_input, aRows, aCols, b_input, bRows, bCols, c_output_gpu);

    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    float gpu_time_milliseconds;
    cudaEventElapsedTime(&gpu_time_milliseconds, start_gpu, stop_gpu);

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

    cudaEvent_t start_cpu, stop_cpu;
    cudaEventCreate(&start_cpu);
    cudaEventCreate(&stop_cpu);
    cudaEventRecord(start_cpu);

    matMulCublas(a_input, aRows, aCols, b_input, bRows, bCols, c_output_cpu);

    cudaEventRecord(stop_cpu);
    cudaEventSynchronize(stop_cpu);
    float cpu_time_milliseconds;
    cudaEventElapsedTime(&cpu_time_milliseconds, start_cpu, stop_cpu);

    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu);

    matMulCUDA(a_input, aRows, aCols, b_input, bRows, bCols, c_output_gpu);

    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    float gpu_time_milliseconds;
    cudaEventElapsedTime(&gpu_time_milliseconds, start_gpu, stop_gpu);

    cout << endl;
    cout << "CUBLAS time: " << cpu_time_milliseconds << " milliseconds" << endl;
    cout << "CUDA time: " << gpu_time_milliseconds << " milliseconds" << endl;
    cout << endl << "Speedup factor: " <<
        cpu_time_milliseconds / gpu_time_milliseconds << endl << endl;

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
    for (int i = 0; i < rows * cols; i++) {
        output[i % cols * rows + i / cols] = input[i];
    }
}


void cudaTransposeTest() {
     std::cout << "------------------------------------------" << std::endl;
    std::cout << "Test Transpose RUNNING." << std::endl;
    const int rows = 770;
    const int cols = 800;

    float *h_input = generateRandomMatrix(rows, cols);
    float *h_output_cpu = (float*) malloc(rows * cols  * sizeof(float));

    Matrix mat;
    mat.dat = h_input;
    mat.rows = rows;
    mat.cols = cols;

    cudaEvent_t start_cpu, stop_cpu;
    cudaEventCreate(&start_cpu);
    cudaEventCreate(&stop_cpu);
    cudaEventRecord(start_cpu);

    transposeCPU(h_input, h_output_cpu, rows, cols);

    cudaEventRecord(stop_cpu);
    cudaEventSynchronize(stop_cpu);
    float cpu_time_milliseconds;
    cudaEventElapsedTime(&cpu_time_milliseconds, start_cpu, stop_cpu);

    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu);

    transposeCUDA(mat, mat);

    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    float gpu_time_milliseconds;
    cudaEventElapsedTime(&gpu_time_milliseconds, start_gpu, stop_gpu);

    cout << endl;
    cout << "CPU time: " << cpu_time_milliseconds << " milliseconds" << endl;
    cout << "GPU time: " << gpu_time_milliseconds << " milliseconds" << endl;
    cout << endl << "Speedup factor: " <<
        cpu_time_milliseconds / gpu_time_milliseconds << endl << endl;

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