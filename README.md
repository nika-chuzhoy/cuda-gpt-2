# Usage
## Setup
`make download` downloads the `vocab.bpe` file needed for this repo to work as well as the checkpoints for `gpt2-124M`, the smallest GPT-2 checkpoint. 
## CPU Demo
`make cpu` runs the interactive CPU demo which allows you to get GPT-2 to autocomplete your text interactively, run entirely on CPU. Note: this demo is non-deterministic, meaning the same input does not produce the same output consistently, as it samples from GPT-2 to allow more variability and quality in output generated. Use `make cpu_seed seed=123` (or any other seed you want) for a deterministic version, such that it can be compared to the outputs of the GPU demo.
## GPU Demo
`make gpu` runs the interactive GPU demo which allows you to get GPT-2 to autocomplete your text interactively, run almost entirely on GPU using the CUDA kernels we have written and integrated. The speedup is very noticeable!!  Note: this demo is non-deterministic, meaning the same input does not produce the same output consistently, as it samples from GPT-2 to allow more variability and quality in output generated. Use `make gpu_seed seed=123` (or any other seed you want) for a deterministic version, such that it can be compared to the outputs of the CPU demo.
## Timed Complete Demo Comparison
`make time` runs a timer script which tests a fixed series of prompts for both GPU and CPU demos with the same fixed seeds, demonstrating their equivalent outputs as well as measuring their times to respond per prompt and in sum, to demonstrate the practical speed up achieved.
## Unit Tests
`make test` runs a series of unit tests comparing CPU functions and their CUDA equivalents, verifying the equivalence of their outputs and the relative speeds.
## Important Note
Note that the demo and timer scripts measure and display approximations of the required and available GPU memory as follows:
```
Available GPU device memory:   12636127232 bytes
Total GPU memory size required: 7247757312 bytes
```
If the total memory required exceeds the memory available, the program will print an error message and crash. This can happen depending on the GPU usage of others on the server, and if it does happen, you can reduce the SEQ_LEN variable in the makefile to reduce the required memory until it is <= the memory available.

# Project Description
The programs run in 2 primary modes. In either case, we inference GPT-2 (the 124M checkpoint, although our program is entirely flexible to larger checkpoints which can be downloaded by modifying the `make download` command in the makefile, because of the fact that larger checkpoints face memory constraints). Either one can provide a fixed prompt which GPT-2 autocompletes (see how `make time` works in the makefile to understand how to use a fixed prompt) until GPT-2 generates the newline token `\n` or one can run the demos and interactively give prompts which are autocompleted by GPT-2 until a `\n` token is generated, at which point one can continue the "conversation" by giving more of a prompt. 2 things to note: 1) we extended the original CPU code credited below to sample tokens from GPT-2 according to the conditional probability distribution from the model's softmax output rather than doing greedy search (arg-max) to choose the most likely token at each step. 2) in exceptionally rare circumstances, the model may get stuck in a loop where it keeps generating tokens without generaing the newline token `\n`, although this is an artifact of GPT-2 itself and not attributable to our implemnentation of its inference. Our big contribution is that we provide the ability to do these 2 forms of inference on the GPU, with inference being done almost entirely (including most expensive computations) on the GPU using CUDA kernels we have written. This provides a visually noticeable speedup during usage.

# Results
<img width="268" alt="image" src="https://github.com/nika-chuzhoy/cuda-gpt-2/assets/68046621/59e4cdd3-d8bf-49a1-a0cc-dd2519d71818">

# Credit

This code is modified from [c-chat-gpt-2](https://github.com/carlini/c-chat-gpt-2/tree/main) by [Nicholas Carlini](https://nicholas.carlini.com/).

# License

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.
