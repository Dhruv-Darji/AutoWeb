import time
from transformers import AutoModelForVision2Seq

model_dir = r"D:\Environments\Models\Qwen2-VL-2B"
print('Attempting CPU load...')
start = time.time()
model = AutoModelForVision2Seq.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True, device_map='cpu')
print('CPU load done, time:', time.time() - start)
print('Model type:', type(model))
