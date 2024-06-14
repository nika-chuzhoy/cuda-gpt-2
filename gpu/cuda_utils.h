#pragma once
#define UNARYdef(fn) Matrix fn##CUDA(Matrix a, float k);
#define UNARY_MTPdef(fn) Matrix fn##CUDA_MTP(Matrix a, float k);
#define BINARYdef(fn) Matrix fn##CUDA(Matrix a, Matrix b);
#define BINARY_MTPdef(fn) Matrix fn##CUDA_MTP(Matrix a, Matrix b);

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float* dat;
    int rows, cols;
} Matrix;

void matMulCUDANaive(float* a, int aRows, int aCols, float* b, int bRows, int bCols, float* out);

void matMulCUDA(float* a, int aRows, int aCols, float* b, int bRows, int bCols, float* out);
void matMulCUDA_MTP(float* a, int aRows, int aCols, float* b, int bRows, int bCols, float* out);

void matMulCublas(float* a, int aRows, int aCols, float* b, int bRows, int bCols, float* out);

void sumCUDA(Matrix a, Matrix out);
// TODO MERGE TEMP
void sumCUDA_MTP(Matrix a, Matrix out);

void transposeCUDA(Matrix a, Matrix out);
void transposeCUDA_MTP(Matrix a, Matrix out); // MERGE TEMP

void softmaxCUDA_MTP(Matrix a);

//Matrix sliceCublas(Matrix a, int b, int rows, int cols);

UNARYdef(divide_const)                    // divide by a constant
UNARYdef(add_const)                       // add a constant
UNARYdef(mat_isqrt)            // square root each entry
UNARYdef(mat_exp)                   // exponetiate each entry
UNARYdef(broadcast)  // copy the first column to every column
UNARYdef(tril)
UNARYdef(GELU)

// TODO MERGE TEMP
UNARY_MTPdef(mtptest)
UNARY_MTPdef(divide_const)                    // divide by a constant
UNARY_MTPdef(add_const)                       // add a constant
UNARY_MTPdef(mat_isqrt)            // square root each entry
UNARY_MTPdef(mat_exp)                   // exponetiate each entry
UNARY_MTPdef(broadcast)  // copy the first column to every column
UNARY_MTPdef(tril)
UNARY_MTPdef(GELU)

BINARYdef(add)       // add two matrices together
BINARYdef(multiply)  // multiply two matrices together
BINARYdef(divide)    // divide the first matrix by the second
BINARYdef(add_tile)
BINARYdef(multiply_tile)

BINARY_MTPdef(add)       // add two matrices together
BINARY_MTPdef(multiply)  // multiply two matrices together
BINARY_MTPdef(divide)    // divide the first matrix by the second
BINARY_MTPdef(add_tile)
BINARY_MTPdef(multiply_tile)

#ifdef __cplusplus
}
#endif
