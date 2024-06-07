#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstdio> 
#include "cuda_utils.h"
#include <time.h>
#include <cuda_runtime.h>

#define LOOP(i, j) for (int i = 0; i < j; i++)

// Unary matrix meta-function here.
// Loop over every entry in a matrix and operate on it
// (independent of any other entry, possibly using some constant k)
#define UNARY(fn, opr)             \
    Matrix fn(Matrix a, float k) { \
        LOOP(i, a.rows* a.cols) {  \
            float b = a.dat[i];    \
            a.dat[i] = opr;        \
        }                          \
        return a;                  \
    }

UNARY(divide_const, b / k)                      // divide by a constant
UNARY(add_const, b + k)                         // add a constant
UNARY(mat_isqrt, 1. / sqrt(b))                  // square root each entry
UNARY(mat_exp, exp(b))                          // exponetiate each entry
UNARY(broadcast, a.dat[(i / a.cols) * a.cols])  // copy the first column to every column

// Tril is the first of two special functions.
//   a   b   c        exp(a/8) exp(b/8) exp(c/8)
//   d   e   f   ->      0     exp(e/8) exp(f/8)
//   g   h   i           0        0        0
// it's use will be described later
UNARY(tril, (i / k < i % (int)k) ? 0 : exp(b / 8))

// GELU is the activation function used for transformers
UNARY(GELU, b / 2 * (1 + tanh(.7978845 * (b + .044715 * b * b * b))))

bool compareMatrices(float *a, float *b, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        if (fabs(a[i] - b[i]) > 1e-2) {
            return false;
        }
    }
    return true;
}

Matrix cloneMatrix(Matrix m) {
    Matrix out;
    out.dat = (float *) malloc(sizeof(float) * m.rows * m.cols);
    out.rows = m.rows;
    out.cols = m.cols;
    for (int i = 0; i < m.rows * m.cols; i++) {
        out.dat[i] = m.dat[i];
    }
    return out;
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

    std::cout << std::endl;
    std::cout << "CPU time: " << cpu_time_milliseconds << " milliseconds" << std::endl;
    std::cout << "GPU time: " << gpu_time_milliseconds << " milliseconds" << std::endl;
    std::cout << std::endl << "Speedup factor: " <<
        cpu_time_milliseconds / gpu_time_milliseconds << std::endl << std::endl;

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

    std::cout << std::endl;
    std::cout << "CUBLAS time: " << cpu_time_milliseconds << " milliseconds" << std::endl;
    std::cout << "CUDA time: " << gpu_time_milliseconds << " milliseconds" << std::endl;
    std::cout << std::endl << "Speedup factor: " <<
        cpu_time_milliseconds / gpu_time_milliseconds << std::endl << std::endl;

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

    std::cout << std::endl;
    std::cout << "CPU time: " << cpu_time_milliseconds << " milliseconds" << std::endl;
    std::cout << "GPU time: " << gpu_time_milliseconds << " milliseconds" << std::endl;
    std::cout << std::endl << "Speedup factor: " <<
        cpu_time_milliseconds / gpu_time_milliseconds << std::endl << std::endl;

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

#define UNARYtest(fn)                                                           \
    void cuda##fn##Test() {                                                     \
        std::cout << "------------------------------------------" << std::endl; \
        std::cout << "Test " << #fn << " RUNNING." << std::endl;                \
        const int rows = 6666;                                                  \
        const int cols = 9999;                                                  \
                                                                                \
        float k = 5.0;                                                          \
                                                                                \
        Matrix cpu_in;                                                          \
        cpu_in.dat = generateRandomMatrix(rows, cols);                          \
        cpu_in.rows = rows;                                                     \
        cpu_in.cols = cols;                                                     \
                                                                                \
        Matrix gpu_in = cloneMatrix(cpu_in);                                    \
                                                                                \
        cudaEvent_t start_cpu, stop_cpu;                                        \
        cudaEventCreate(&start_cpu);                                            \
        cudaEventCreate(&stop_cpu);                                             \
        cudaEventRecord(start_cpu);                                             \
                                                                                \
        Matrix cpu_out = fn(cpu_in, k);                                         \
                                                                                \
        cudaEventRecord(stop_cpu);                                              \
        cudaEventSynchronize(stop_cpu);                                         \
        float cpu_time_milliseconds;                                            \
        cudaEventElapsedTime(&cpu_time_milliseconds, start_cpu, stop_cpu);      \
                                                                                \
        cudaEvent_t start_gpu, stop_gpu;                                        \
        cudaEventCreate(&start_gpu);                                            \
        cudaEventCreate(&stop_gpu);                                             \
        cudaEventRecord(start_gpu);                                             \
                                                                                \
        Matrix gpu_out = fn##CUDA(gpu_in, k);                                   \
                                                                                \
        cudaEventRecord(stop_gpu);                                              \
        cudaEventSynchronize(stop_gpu);                                         \
        float gpu_time_milliseconds;                                            \
        cudaEventElapsedTime(&gpu_time_milliseconds, start_gpu, stop_gpu);      \
                                                                                \
        std::cout << std::endl;                                                 \
        std::cout << "CPU time: " << cpu_time_milliseconds << " milliseconds" << std::endl;\
        std::cout << "GPU time: " << gpu_time_milliseconds << " milliseconds" << std::endl;\
        std::cout << std::endl << "Speedup factor: " <<                         \
            cpu_time_milliseconds / gpu_time_milliseconds << std::endl << std::endl;\
                                                                                \
        if (compareMatrices(cpu_out.dat, gpu_out.dat, cols, rows)) {            \
            std::cout << "Test " << #fn << " PASSED." << std::endl;             \
        } else {                                                                \
            std::cout << "Test " << #fn << " FAILED." << std::endl;             \
        }                                                                       \
                                                                                \
        free(cpu_in.dat);                                                       \
        free(gpu_in.dat);                                                       \
    }

UNARYtest(divide_const)                     
UNARYtest(add_const)                        
UNARYtest(mat_isqrt)                 
UNARYtest(mat_exp)                         
UNARYtest(broadcast)
UNARYtest(tril)
UNARYtest(GELU)

int main() {

    matMulCUDATest();
    matMulCublasTest();
    cudaTransposeTest();
    cudadivide_constTest();
    cudaadd_constTest();   
    cudamat_isqrtTest(); 
    cudamat_expTest();                       
    cudabroadcastTest();
    cudatrilTest();
    cudaGELUTest();

    return 0;
}
