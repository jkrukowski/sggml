# sggml

Attempt to use [ggml](https://github.com/ggerganov/ggml) from Swift, mostly for the learning purposes. For now it 
contains only (example taken from the original repo):
- MNIST
- GPT-2

Usage:

### MNIST

```bash
# Build and run with the random MNIST image
$ swift run mnist "/path/to/mnist/model/ggml-model-f32.bin" "/path/to/images/file/t10k-images.idx3-ubyte"
```

### GPT-2

```bash
# Build and run with the random MNIST image
$ swift run gpt2 "/path/to/gpt2/model/ggml-model-f32.bin" "Prompt text"
```
