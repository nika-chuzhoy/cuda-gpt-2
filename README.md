# Usage
```
# Download model
make download

# Run CPU Demo
make cpu

# Run GPU demo
make gpu

# Run CPU demo with fixed seed
make cpu_seed seed=123

# Run GPU demo with fixed seed
make gpu_seed seed=123

# Run unit tests
make test
```

# Credit

This code is modified from [c-chat-gpt-2](https://github.com/carlini/c-chat-gpt-2/tree/main) by [Nicholas Carlini](https://nicholas.carlini.com/).
We extend this project in the following ways:
* Instead of greedy search (arg-max), we sample according to the conditional probability distribution from the model's softmax output.
* We implement GPT-2 inference entirely in CUDA. 

# License

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.