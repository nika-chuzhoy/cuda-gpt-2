.PHONY: all cpu gpu download clean

# Paths
CPU_SRC = cpu/c_chat_gpt_2.c
CPU_BIN = bin/c_chat_gpt_2

GPU_SRC_C = gpu/optimized_chat_gpt_2.c
GPU_SRC_CU = gpu/cuda_utils.cu
GPU_OBJ = bin/cuda_utils.o
GPU_BIN = bin/optimized_chat_gpt_2

TEST_SRC = gpu/test.cpp
TEST_BIN = bin/test_matmul

# Common variables
SEQ_LEN = 512

# Targets
cpu: clean
	gcc -O3 $(CPU_SRC) -lm -o $(CPU_BIN)
	./bin/c_chat_gpt_2 gpt2-124M.ckpt vocab.bpe "$$(echo -e "\nAlice: Hello, how are you doing today?\nBob: I am doing well. I am a language model trained by OpenAI. How can I assist you?\nAlice: Can you answer my questions?\nBob: Yes I will answer your questions. What do you want to know?\nAlice: What is your name?\nBob: My name is Bob.\nAlice: Nice to meet you Bob. I'm alice.\nBob: How can I help you?")" 512

gpu: clean
	nvcc -c $(GPU_SRC_CU) -o $(GPU_OBJ)
	gcc -O3 $(GPU_SRC_C) $(GPU_OBJ) -o $(GPU_BIN) -L/usr/local/cuda/lib64 -lcudart -lcublas -lm -lstdc++ 
	./bin/optimized_chat_gpt_2 gpt2-124M.ckpt vocab.bpe "$$(echo -e "\nAlice: Hello, how are you doing today?\nBob: I am doing well. I am a language model trained by OpenAI. How can I assist you?\nAlice: Can you answer my questions?\nBob: Yes I will answer your questions. What do you want to know?\nAlice: What is your name?\nBob: My name is Bob.\nAlice: Nice to meet you Bob. I'm alice.\nBob: How can I help you?")" 512

test: clean
	nvcc -c $(GPU_SRC_CU) -o $(GPU_OBJ)
	gcc -O3 $(TEST_SRC) $(GPU_OBJ) -o $(TEST_BIN) -L/usr/local/cuda/lib64 -lcudart -lcublas -lm -lstdc++

download:
	curl -o vocab.bpe https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe
	curl -o gpt2-124M.ckpt https://openaipublic.blob.core.windows.net/gpt-2/models/124M/model.ckpt.data-00000-of-00001
	# Uncomment the following lines to download other model checkpoints
	# curl -o gpt2-355M.ckpt https://openaipublic.blob.core.windows.net/gpt-2/models/355M/model.ckpt.data-00000-of-00001
	# curl -o gpt2-774M.ckpt https://openaipublic.blob.core.windows.net/gpt-2/models/774M/model.ckpt.data-00000-of-00001
	# curl -o gpt2-1558M.ckpt https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/model.ckpt.data-00000-of-00001

clean:
	rm -f $(GPU_OBJ) $(CPU_BIN) $(GPU_BIN)
