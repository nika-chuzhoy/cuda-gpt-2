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

void cudaTranspose(Matrix a);

#ifdef __cplusplus
}
#endif
