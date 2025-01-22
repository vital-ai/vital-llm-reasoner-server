from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

quantized_model_path = "./qwen-quantized"


# file_path = "qwen-quantized/model.safetensors"

file_path = "/Users/hadfield/Desktop/model.safetensors"


try:
    with safe_open(file_path, framework="pt", device="cpu") as f:
        keys = f.keys()
        print(f"Keys in the safetensors file: {keys}")
        print(f"Total tensors: {len(keys)}")
except Exception as e:
    print(f"Error: {e}")

exit(0)

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)

print("Loading quantized model for test inference...")
model = AutoModelForCausalLM.from_pretrained(
    quantized_model_path,
    # torch_dtype=torch.qint8,
    # low_cpu_mem_usage=True,
    # device_map="auto"
)

print("Loaded quantized model for test inference...")

exit(0)

prompt = "What is the meaning of life?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(inputs["input_ids"], max_length=50)
print("Model output:", tokenizer.decode(outputs[0], skip_special_tokens=True))

"""
# Verify the quantized shards
print("Verifying quantized model shards...")
model_shards = [
    os.path.join(quantized_model_path, f)
    for f in sorted(os.listdir(quantized_model_path))
    if f.endswith(".safetensors")
]
if not model_shards:
    raise ValueError("No quantized shards found in the directory!")

try:
    for shard in model_shards[:1]:  # Test the first shard
        print(f"Checking shard: {shard}")
        quantized_tensors = torch.load(shard, map_location="cpu")
        print(f"Shard tensors loaded: {list(quantized_tensors.keys())[:5]} (truncated)")
except Exception as e:
    print(f"Error loading shard: {e}")
    raise
    """