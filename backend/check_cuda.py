import torch
print("CUDA Version:", torch.version.cuda)
print("PyTorch Version:", torch.__version__)
print("Is CUDA available?", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())