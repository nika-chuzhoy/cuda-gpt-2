/* c_chat_gpt_2.c: a minimal gpt-2 implementation in ~3000 bytes of C
 * Copyright (C) 2023 Nicholas Carlini.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include"cuda_utils.h"

int DIM, NLAYER, NHEAD;

int token_processed_upto;
int num_total_tokens;
int tmp, zz;
char* bpe;

void *memory, *memory_top;
FILE* fp;

Matrix* layer_weights;

// Standard stuff here. Let's save space with all our loops
#define LOOP(i, j) for (int i = 0; i < j; i++)

// A matrix is just a 2d vector of floats with rows and columns.
Matrix NewMatrix(int rows, int cols, int reuse) {
    float* a = memory;
    memory += tmp = 4 * rows * cols;
    memset(a, 0, tmp * reuse);
    Matrix out = {a, rows, cols};
    return out;
}

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

// Binary matrix meta-function here.
// Loop over pairs of entries in two matricies and operate on them
#define BINARY(fn, opr)                                               \
    Matrix fn(Matrix a, Matrix b) {                                   \
        LOOP(i, a.rows* a.cols) { a.dat[i] = a.dat[i] opr b.dat[i]; } \
        return a;                                                     \
    }

BINARY(add, +)       // add two matrices together
BINARY(multiply, *)  // multiply two matrices together
BINARY(divide, /)    // divide the first matrix by the second
BINARY(subtract, -)

// We also have some ugly hacks here to implement "tiling"
// that lets us add or multiply one matrix by the first column of a second
// To do this tiling, we don't want to operate on b.dat[i], so instead
// we re-index with what we want and then just stick a ; there to
// drop the actual b.dat[i]
BINARY(add_tile, +b.dat[i % a.cols];)
BINARY(multiply_tile, *b.dat[i % a.cols];)

// Compute the sum of the rows in a matrix, populating each row with the same sum
Matrix sum(Matrix a) {
    Matrix out = NewMatrix(a.rows, a.cols, 1);

    LOOP(i, a.rows * a.cols) {
        out.dat[(i / a.cols) * a.cols] += a.dat[i];
    }

    broadcast(out, 0);
    return out;
}

// Transpose a matrix flipping the rows and columns
Matrix transpose(Matrix a) {
    Matrix out = NewMatrix(a.cols, a.rows, 1);
    transposeCUDA(a, out);
    return out;
}

// Efficient incremental matrix multiplication.
// We make the following optimizations:
// 1. Instead of multiplying A by B, we do A by transpose(B)
//    This keeps the reads out of the B matrix in sequential order
//    which helps cache efficiency
// 2. Instaed of performing the product all at once, we block it
//    into 4x4 inner computations which again is much more cache efficient
// 3. If the fast flag is defined, we use OMP to parallelize across threads
// 4. We re-use computation from prior runs, and only fill in the
//    *new* rows that weren't populated the prior run through the model
void printMatrix(Matrix m) {
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            printf("%f ", m.dat[i * m.cols + j]);
        }
        printf("\n");
    }
}

Matrix matmul_t_fast(Matrix a, Matrix b) {
  Matrix out = NewMatrix(a.rows, b.rows, !token_processed_upto);

  // Use the CUDA matrix multiplication function
  matMulCUDA(a.dat + token_processed_upto * a.cols, num_total_tokens - token_processed_upto, a.cols, b.dat, b.rows, b.cols, out.dat + token_processed_upto * b.rows);

  return add(NewMatrix(out.rows, out.cols, 1), out);
}

// Take a slice out of a larger matrix and return a new matrix with the given shape
Matrix slice(Matrix a, int b, int rows, int cols) {
    Matrix out = {a.dat + b * rows, rows, cols};
    return out;
}

// A somewhat weird unary operator that computes the "layernorm" operator.
// Exactly what it does doesn't matter.
Matrix LayerNorm(Matrix a, int i) {
    Matrix b = add(a, divide_const(sum(a), -a.cols));
    Matrix k = divide_const(sum(multiply(add(NewMatrix(b.rows, b.cols, 1), b), b)), b.cols - 1);  // todo can remove -1
    Matrix out = add_tile(multiply_tile(multiply(add(NewMatrix(b.rows, b.cols, 1), b), mat_isqrt(add_const(k, 1e-5), 0)), layer_weights[i + 1]), layer_weights[i]);

    return out;
}

// Compute a linear matrix layer, x * W + b
#define Linear(a, i) add_tile(matmul_t_fast(a, layer_weights[i + 1]), layer_weights[i])

// Read a weight matrix out of the data file into memory
Matrix read_matrix(int rows, int cols) {
    rows += !rows;  // if rows == 0 then load at least one row
    cols += !cols;  // if cols == 0 then load at least one col

    Matrix a = NewMatrix(rows, cols, 1);

    // It's already stored as a float on disk. Just load the bytes.
    // (This assumes your machine is little endian)
    size_t result = fread(a.dat, tmp, 1, fp);
    if (result != 1) {
        // Handle error, e.g., print an error message and exit
        perror("Error reading matrix");
        exit(EXIT_FAILURE);
    }

    // Our matrix multiply assumes transposed weights.
    return transpose(a);
}

// And now for something completely different: byte pair encoding
// This function takes a single word and produces the tokenization of that word
// We do this with an exponential-time algorithm that's very short:
// for every token we know about, see if it fits in the first position.
// If it does, see what the cost of tokenizing the rest of the word would be.
// Return the first token that has the lowest overall cost.
int best_outi;
int fix(char* out) {
    if (!*out) return 0;
    int result = 1e9;
    int best_i;
    LOOP(i, 5e4) {
        // if 1st char of the token is not \0 and if 1st part of out matches token
        if (bpe[999 * i] && strncmp(bpe + 999 * i, out, tmp = strlen(bpe + 999 * i)) == 0) {
            int sub_cost = fix(out + tmp) + i + 1e7;
            if (sub_cost < result) {
                result = sub_cost;
                best_i = i;
            }
        }
    }
    best_outi = best_i;
    return result;
}

// Given the ability to byte-pair encode a single word, this encodes a sentence
// by splitting it into individual words, and tokenizing each word separately
int* tokenize(char* seq, /*INT*/ int* result) {
    char out[1000];
    int i = 0;
    while (seq[i]) {
        int j = i++;
        while (47 < seq[i] && seq[i] < 58 || 64 < seq[i]) {
            fflush(stdout);
            i++;
        }
        strcpy(out, seq + j);
        out[i - j] = 0;
        fflush(stdout);
        int k = 0;
        while (out[k]) {
            fix(out + k);
            char* ntok = bpe + best_outi * 999;
            k += strlen(ntok);
            *result++ = best_outi;
        }
    }
    return result;
}

void do_inference(clock_t start, clock_t end, double cpu_time_used, Matrix wpe, Matrix wte, Matrix *weights, int T, char *buf, int *output){
    num_total_tokens = tokenize(buf, output) - output;

    memory_top = memory;

    token_processed_upto = 0;

    while (1) {  // Brian loop
        // Reset the memory to the top of the original value
        memory = memory_top;

        // Compute the context window size as the next largest multiple of 32
        T = num_total_tokens + 32 - num_total_tokens % 32;
        // If the number is 0 mod 32, then we need to recompute everything bottom up
        token_processed_upto *= !!(num_total_tokens % 32);

        // This is the line we're going to process.
        Matrix line = NewMatrix(T, DIM, 1);

        // Start by loading the embedding weights and adding the position encoding.
        LOOP(i, num_total_tokens) {
            LOOP(j, DIM) {
                line.dat[i * DIM + j] = wte.dat[output[i] * DIM + j] + wpe.dat[j * 1024 + i];
            }
        }

        // Start the transformer neural network inference.
        LOOP(i, NLAYER) {  // Lynn loop
            // The layers on disk are stored by sorting alphabetically,
            // because tensorflow makes no sense. We need to convert this to
            // the correct order. For example, if there are 12 layers, we would
            // have them on disk in order: 0 1 10 11 2 3 4 5 6 7 8 9
            // which means we permute by the inverse: 0 1 4 5 6 7 8 9 10 11 2 3
            int permute;
            tmp = 0;
            LOOP(j, 10) {
                if (j == i) {
                    permute = tmp;
                }
                tmp++;
                LOOP(k, 10 * (j > 0)) {
                    if (j * 10 + k < NLAYER && tmp++ && i == j * 10 + k) {
                        permute = tmp;
                    }
                }
            }

            // This layer's weights are at this offset
            layer_weights = weights + 12 * permute;

            // Compute the keys, queries, and values all at once with a big multiply
            Matrix qkv = transpose(slice(Linear(LayerNorm(line, 4), 0), 0, T * 3, DIM));

            // Make space for the output of the computation
            Matrix result = NewMatrix(DIM, T, 1);

            LOOP(k, NHEAD) {
                // Split the qkv into each of the heads
                Matrix merge = transpose(slice(qkv, k * 3, 64 * T, 3)),
                    // perform the product of the queries and keys and then exponentiate
                    a = tril(matmul_t_fast(transpose(slice(merge, 0, 64, T)),
                                        transpose(slice(merge, T, 64, T))),
                            T),
                    // finally multiply the softmax output (a/sum(a)) with the values matrix
                    out = transpose(matmul_t_fast(divide(a, sum(a)), slice(merge, T * 2, 64, T)));
                // and copy the output to the proper location in the result matrix
                memcpy(result.dat + 64 * T * k, out.dat, 64 * T * 4);
            }

            // Residual connection
            line = add(line, Linear(transpose(result), 2));

            // Activation function and residual connection
            line = add(line, Linear(GELU(Linear(LayerNorm(line, 6), 8), 0), 10));
        }

        // Reset layer weights so we can do the last layer norm
        layer_weights = weights;
        line = LayerNorm(line, 12 * NLAYER);

        // And finally compute the output logits
        token_processed_upto = 0;
        int tmp = num_total_tokens;
        num_total_tokens = 1;
        Matrix result = matmul_t_fast(transpose(slice(line, tmp - 1, DIM, 1)), wte);
        token_processed_upto = num_total_tokens = tmp;

        // Calculate softmax probabilities
        int size = 5e4;
        float temperature = 0.7;
        float* logits = divide_const(result, temperature).dat;
        double max = logits[0];
        for (int i = 1; i < size; i++) {
            if (logits[i] > max) {
                max = logits[i];
            }
        }

        double sum = 0.0;
        double probs[size];
        for (int i = 0; i < size; i++) {
            probs[i] = exp(logits[i] - max);
            sum += probs[i];
        }

        for (int i = 0; i < size; i++) {
            probs[i] /= sum;
        }

        // Weighted random sampling
        tmp = 0;
        float cumulative[size];
        cumulative[0] = probs[0];
        for (int i = 1; i < size; i++) {
            cumulative[i] = cumulative[i - 1] + probs[i];
        }

        float r = ((float)rand() / RAND_MAX) * cumulative[size - 1];

        for (int i = 0; i < size; i++) {
            if (r < cumulative[i]) {
                tmp = i;
                break;
            }
        }

        // If the history is too long, then purge by half
        if (num_total_tokens == zz) {
            memcpy(output, output + zz / 2, tmp * 2);
            num_total_tokens -= zz / 2;
            token_processed_upto = 0;
        }
        // Write it to the history buffer
        output[num_total_tokens++] = tmp;

        // If it's a newline this is the end of the converstaion
        if (bpe[tmp * 999] == 10) {
            end = clock();
            cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
            printf("\n\n----Seconds to respond: %f----\n", cpu_time_used);
            break;
        }

        // Otherwise print it and keep generating along
        printf("%s", bpe + tmp * 999);
        fflush(stdout);
    }
}

// Now for the main function that does most of the useful work.
int main(int tmp, char** argv) {
    clock_t start, end;
    double cpu_time_used;
    start = clock();
    bool is_set_prompt = false;
    char *set_prompt;
    bool is_set_seed = false;
    int seed;

    //  Set random seed, for testing purposes
    if (tmp >= 5) {
        seed = atoi(argv[4]);
        is_set_seed = true;
    }
    //  If this is a set-prompt run
    if (tmp >= 6) {
        set_prompt = argv[5];
        is_set_prompt = true;
    }

    // Initially let's figure out the right hyperparameters for this model
    // argv[1] stores the name of the model we're loading
    // tmp will map 124M -> 0, 355M -> 1, 775M -> 2, 1558M -> 3
    // Note that if you change the name of the file then this will break.
    tmp = argv[1][5] + 3 * argv[1][7] + 3 & 3;
    // Now we just compute the layer sizes from tmp
    NHEAD = 12 + 4 * tmp + (tmp > 2);
    DIM = NHEAD * 64;
    NLAYER = 12 * tmp + 12;

    // Allocate space
    zz = atoi(argv[3]);
    memory = malloc(2LL * DIM * DIM * NLAYER * zz);

    /////////////////////////////////////////////////////////////
    ////////////////LOAD BPE FUNCTION INLINED////////////////////
    /////////////////////////////////////////////////////////////
    bpe = malloc(1e9);

    // load the bpe file from argv[2]
    fp = fopen(argv[2], "r");

    // The BPE was not written in a c-friendly format.
    // So we need to do some ugly processing to load it.
    unsigned char a[tmp = 999], b[tmp];
    LOOP(i, 5e4) {  // Vansh loop
        int k = i * tmp;
        if (i < 93) {
            // The first 92 tokens are just the printable ascii characters
            bpe[k] = i + 33;
            bpe[k + 1] = 0;
        } else if (i > 254) {
            // Ones above 254 are from the BPE file. Load those
            int fscanf_result = fscanf(fp, "%s %s", a, b);
            if (fscanf_result != 2) {
                // Handle error, e.g., print an error message and exit
                perror("Error reading BPE");
                exit(EXIT_FAILURE);
            }

            strcat((char*)a, (char*)b);
            int j = 0;
            LOOP(i, a[i]) {
                // UTF8 encoding makes life hard so handle that here
                bpe[k + j++] = a[i] ^ 196 ? a[i] : a[++i] - 128;
            }
            bpe[k + j++] = 0;
        } else if (i > 187) {
            // Tokens above 187 are the nonprintable asii character from 0-32
            bpe[k] = i - 188;
            bpe[k + 1] = 0;
        }
    }

    fp = fopen(argv[1], "r");

    /////////////////////////////////////////////////////////////
    //////////////READ MATRIX FUNCTION INLINED///////////////////
    /////////////////////////////////////////////////////////////
    Matrix weights[999];
    Matrix* out = weights;

    LOOP(i, NLAYER){
        LOOP(j, 12){
                // These two nasty expressions compute the shapes of the matricies on disk
                * out++ = read_matrix(DIM + DIM * (j ? j ^ 8 ? j ^ 11 ? 0 : 3 : 3 : 2), DIM * ((j % 8 == 3) + 3 * (j % 8 == 1) + (j == 9)));
        }
    }

    *out++ = read_matrix(DIM, 1);  // ln_f.bias
    *out++ = read_matrix(DIM, 1);  // ln_f.weight

    Matrix wpe = read_matrix(1024, DIM),
        wte = transpose(read_matrix(5e4, DIM));

    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("\n---- Seconds to load: %f ----\t\n", cpu_time_used);

    /////////////////////////////////////////////////////////////
    ///////////////INFERENCE FUNCTION INLINED////////////////////
    /////////////////////////////////////////////////////////////

    if(is_set_prompt) {
        start = clock();
        srand(seed);

        char buf[1000] = {0};
        int T;
        printf("\nHuman: ");
        printf("%s\n", set_prompt);
        fflush(stdout);

        strcpy(buf, set_prompt);

        // This is going to store our prompt
        int output[2 * zz];
        num_total_tokens = 0;

        printf("AI: ");
        strcat(buf, "\n\n");

        do_inference(start, end, cpu_time_used, wpe, wte, weights, T, buf, output);
    } else {
        while (1) {  // Nika loop
            start = clock();
            srand(seed);

            char buf[1000] = {0};
            int T;
            printf("\nHuman: ");
            fflush(stdout);

            char* fgets_result = fgets(buf, 1000, stdin);
            if (fgets_result == NULL) {
                // Handle error, e.g., print an error message and exit
                perror("Error reading input");
                exit(EXIT_FAILURE);
            }

            // This is going to store our prompt
            int output[2 * zz];
            num_total_tokens = 0;

            printf("AI: ");
            strcat(buf, "\n\n");

            do_inference(start, end, cpu_time_used, wpe, wte, weights, T, buf, output);
        }
    }
}
