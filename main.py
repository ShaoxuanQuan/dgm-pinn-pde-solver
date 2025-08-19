import torch

# 检查CUDA是否可用
cuda_available = torch.cuda.is_available()
print(f"Is CUDA available? {cuda_available}")