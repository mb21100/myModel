import torch
print("Torch version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("Current Device:", torch.cuda.current_device())
