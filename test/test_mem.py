import argparse
import os
import sys
sys.path.append("../")
import torch
from model.pvt import PVTFormer
from torchmeter import Meter


img = torch.randn(1, 1, 416, 416).cuda()
# Measure GPU memory usage
torch.cuda.empty_cache()
start_memory = torch.cuda.memory_allocated()
# measure the memory cost of initializing our model. 
with torch.no_grad():
    model = PVTFormer(in_channels=1).cuda()
    model.eval()
    model_output = model(img)
    torch.cuda.synchronize()
end_memory = torch.cuda.memory_allocated()
total_memory_reserved = torch.cuda.memory_reserved()

print(f"Memory allocated before model run: {start_memory / (1024 ** 3):.2f} GiB")
print(f"Memory allocated after model run: {end_memory / (1024 ** 3):.2f} GiB")
print(f"Total memory reserved by PyTorch: {total_memory_reserved / (1024 ** 3):.2f} GiB")
print(f"Memory used by the model: {(end_memory - start_memory) / (1024 ** 3):.2f} GiB")

print("**********************************************************************************")

print(f"Memory allocated before model run: {start_memory / (1024 ** 2):.2f} MiB")
print(f"Memory allocated after model run: {end_memory / (1024 ** 2):.2f} MiB")
print(f"Total memory reserved by PyTorch: {total_memory_reserved / (1024 ** 2):.2f} MiB")
print(f"Memory used by the model: {(end_memory - start_memory) / (1024 ** 2):.2f} MiB")
