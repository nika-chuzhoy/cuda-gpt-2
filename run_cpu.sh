#!/bin/bash

# Compile C code
gcc -O3 cpu/c_chat_gpt_2.c -lm -o cpu/c_chat_gpt_2

# Run the program
./cpu/c_chat_gpt_2 gpt2-124M.ckpt vocab.bpe 1024