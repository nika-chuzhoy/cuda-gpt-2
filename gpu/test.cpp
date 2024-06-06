#include <cstdio> 
#include "cuda_utils.h"

// Define matrices A and B
const int aRows = 5;
const int aCols = 2;
const int bRows = 4;
const int bCols = 2;

float A[aRows * aCols] = {1, 2,
                        3, 4,
                        5, 6,
                        7, 8,
                        9, 10};

float B[bRows * bCols] = {0, 1, 
                        10, 1,
                        0, 1, 
                        0, 1};

void matMulCUDATest() {
    float C[aRows * bRows];
    matMulCublas(A, aRows, aCols, B, bRows, bCols, C);

    printf("CUBLAS Matrix:\n");
    for (int i = 0; i < aRows; ++i) {
        for (int j = 0; j < bRows; ++j) {
            printf("%.2f ", C[i * bRows + j]);
        }
        printf("\n");
    }
}

void matMulCublasTest() {
    float C[aRows * bRows];
    matMulCUDA(A, aRows, aCols, B, bRows, bCols, C);

    printf("CUDA Matrix:\n");
    for (int i = 0; i < aRows; ++i) {
        for (int j = 0; j < bRows; ++j) {
            printf("%.2f ", C[i * bRows + j]);
        }
        printf("\n");
    }
}

void sliceCublasTest() {
    int cRows = 1;
    int cCols = 1;
    int b = 0;
    Matrix C = sliceCublas({A, aRows, aCols}, b, cRows, cCols);

    printf("Sliced Matrix:\n");
    for (int i = 0; i < cRows; ++i) {
        for (int j = 0; j < cCols; ++j) {
            printf("%.2f ", C.dat[i * cCols + j]);
        }
        printf("\n");
    }
}

int main() {

    //matMulCUDATest();
    //matMulCublasTest();
    sliceCublasTest();

    return 0;
}