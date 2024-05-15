#ifndef MATMUL_CUDA_H
#define MATMUL_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

void matMulCUDA(float* a, int aRows, int aCols, float* b, int bRows, int bCols, float* out);

#ifdef __cplusplus
}
#endif

#endif
