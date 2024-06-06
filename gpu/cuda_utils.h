#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float* dat;
    int rows, cols;
} Matrix;

void matMulCUDA(float* a, int aRows, int aCols, float* b, int bRows, int bCols, float* out);

void matMulCublas(float* a, int aRows, int aCols, float* b, int bRows, int bCols, float* out);

void transposeCUDA(Matrix a);

Matrix sliceCublas(Matrix a, int b, int rows, int cols);

#ifdef __cplusplus
}
#endif
