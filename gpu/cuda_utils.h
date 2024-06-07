#pragma once
#define UNARYdef(fn) Matrix fn##CUDA(Matrix a, float k);

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float* dat;
    int rows, cols;
} Matrix;

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

#ifdef __cplusplus
}
#endif
