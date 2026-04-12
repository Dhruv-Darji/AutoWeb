import time
import torch
from transformers import AutoModelForVision2Seq

print('CUDA available:', torch.cuda.is_available())
start = time.time()
print('Starting GPU load...')
model = AutoModelForVision2Seq.from_pretrained('D:/Environments/Models/Qwen2-VL-2B', local_files_only=True, trust_remote_code=True, device_map='auto', torch_dtype=torch.float16)
print('GPU load done, time:', time.time() - start)
print('Model device:', next(model.parameters()).device)
