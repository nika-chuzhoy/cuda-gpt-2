#!/bin/bash

# Compile CUDA code
nvcc -c gpu/cuda_utils.cu -o gpu/cuda_utils.o

# Compile C code and link with CUDA object file
gcc -O3 gpu/optimized_chat_gpt_2.c gpu/cuda_utils.o -o gpu/optimized_chat_gpt_2 -L/usr/local/cuda/lib64 -lcudart -lm -lstdc++

# Run the program
./gpu/optimized_chat_gpt_2 gpt2-124M.ckpt vocab.bpe "$(echo -e "\nAlice: Hello, how are you doing today?\nBob: I am doing well. I am a language model trained by OpenAI. How can I assist you?\nAlice: Can you answer my questions?\nBob: Yes I will answer your questions. What do you want to know?\nAlice: What is your name?\nBob: My name is Bob.\nAlice: Nice to meet you Bob. I'm alice.\nBob: How can I help you?")" 512