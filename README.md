# sggml

Attempt to use [ggml](https://github.com/ggerganov/ggml) from Swift, mostly for the learning purposes. For now it 
contains only (example taken from the original repo):
- MNIST
- GPT-2

Usage:

### MNIST

```bash
# Build and run MNIST with a random image
$ swift run mnist "/path/to/mnist/model/ggml-model.bin" "/path/to/images/file/t10k-images.idx3-ubyte"
```

### GPT-2

```bash
# Build and run GPT-2 with a `Prompt text` prompt
$ swift run gpt2 "/path/to/gpt2/model/ggml-model.bin" "Prompt text"
```

### Bert Spam

```bash
# Build and run Bert Spam detection with a `Prompt text` prompt
$ swift run bert-spam "/path/to/gpt2/model/ggml-model.bin" "Prompt text"
```
