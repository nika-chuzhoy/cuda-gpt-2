.PHONY: all cpu gpu download clean

# Paths
CPU_SRC = cpu/c_chat_gpt_2.c
CPU_BIN = bin/c_chat_gpt_2

GPU_SRC_C = gpu/optimized_chat_gpt_2.c
GPU_SRC_CU = gpu/cuda_utils.cu
GPU_OBJ = bin/cuda_utils.o
GPU_BIN = bin/optimized_chat_gpt_2

SEQ_LEN = 1024

# Targets
all: cpu gpu

cpu: bin
	gcc -O3 $(CPU_SRC) -lm -o $(CPU_BIN)
	./bin/c_chat_gpt_2 gpt2-124M.ckpt vocab.bpe $(SEQ_LEN)

gpu: bin
	nvcc -c $(GPU_SRC_CU) -o $(GPU_OBJ)
	gcc -O3 $(GPU_SRC_C) $(GPU_OBJ) -o $(GPU_BIN) -L/usr/local/cuda/lib64 -lcudart -lm -lstdc++
	./bin/optimized_chat_gpt_2 gpt2-124M.ckpt vocab.bpe $(SEQ_LEN)

download:
	curl -o vocab.bpe https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe
	curl -o gpt2-124M.ckpt https://openaipublic.blob.core.windows.net/gpt-2/models/124M/model.ckpt.data-00000-of-00001
	# Uncomment the following lines to download other model checkpoints
	# curl -o gpt2-355M.ckpt https://openaipublic.blob.core.windows.net/gpt-2/models/355M/model.ckpt.data-00000-of-00001
	# curl -o gpt2-774M.ckpt https://openaipublic.blob.core.windows.net/gpt-2/models/774M/model.ckpt.data-00000-of-00001
	# curl -o gpt2-1558M.ckpt https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/model.ckpt.data-00000-of-00001

clean:
	rm -f $(GPU_OBJ) $(CPU_BIN) $(GPU_BIN)

bin:
	mkdir -p bin