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

# Credit

This code is modified from [c-chat-gpt-2](https://github.com/carlini/c-chat-gpt-2/tree/main) by [Nicholas Carlini](https://nicholas.carlini.com/).
We extend this project in the following ways:
* Instead of greedy search (arg-max), we sample according to the conditional probability distribution from the model's softmax output.
* We implement GPT-2 inference entirely in CUDA. 

# License

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.
