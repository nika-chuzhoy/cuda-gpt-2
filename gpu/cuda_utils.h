#pragma once
#define UNARYdef(fn) Matrix fn##CUDA(Matrix a, float k);
#define BINARYdef(fn) Matrix fn##CUDA(Matrix a, Matrix b);

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float* dat;
    int rows, cols;
} Matrix;

void matMulCUDANaive(float* a, int aRows, int aCols, float* b, int bRows, int bCols, float* out);

void matMulCUDA(float* a, int aRows, int aCols, float* b, int bRows, int bCols, float* out);

void matMulCublas(float* a, int aRows, int aCols, float* b, int bRows, int bCols, float* out);

void transposeCUDA(Matrix a, Matrix out);

Matrix sliceCublas(Matrix a, int b, int rows, int cols);

UNARYdef(divide_const)                    // divide by a constant
UNARYdef(add_const)                       // add a constant
UNARYdef(mat_isqrt)            // square root each entry
UNARYdef(mat_exp)                   // exponetiate each entry
UNARYdef(broadcast)  // copy the first column to every column
UNARYdef(tril)
UNARYdef(GELU)

BINARYdef(add)       // add two matrices together
BINARYdef(multiply)  // multiply two matrices together
BINARYdef(divide)    // divide the first matrix by the second
BINARYdef(add_tile)
BINARYdef(multiply_tile)

#ifdef __cplusplus
}
#endif
