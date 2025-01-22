import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from safetensors.torch import safe_open, save_file


"""
model = AutoModelForCausalLM.from_pretrained(local_model_path, torch_dtype="auto")

for name, param in model.named_parameters():
    print(f"Parameter: {name}, Data Type: {param.dtype}")

# exit(0)
"""


local_model_path = "/Users/hadfield/Local/huggingface-git/QwQ-32B-Preview"
output_dir = "qwen-quantized"

print("Loading the tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
tokenizer.save_pretrained(output_dir)

# Supported data types for processing
SUPPORTED_DTYPES = {torch.float32, torch.bfloat16, torch.int8}

# Process and quantize shards
quantized_tensors = {}
shard_dir = local_model_path
shards_processed = 0
tensors_skipped = 0

for shard_file in sorted(os.listdir(shard_dir)):
    if not shard_file.endswith(".safetensors"):
        continue

    shard_path = os.path.join(shard_dir, shard_file)
    print(f"Processing shard: {shard_file}")

    try:
        # Load shard using safetensors
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            for key in shard.keys():
                tensor = shard.get_tensor(key)
                print(f"Processing tensor: {key}, dtype: {tensor.dtype}, shape: {tensor.shape}")

                # Log statistics of the tensor
                print(f"  Num elements: {tensor.numel()}")
                print(f"  Min: {tensor.min().item() if tensor.numel() > 0 else 'N/A'}")
                print(f"  Max: {tensor.max().item() if tensor.numel() > 0 else 'N/A'}")
                print(f"  Mean: {tensor.mean().item() if tensor.numel() > 0 else 'N/A'}")
                print("-" * 50)

                # Validate tensor data type
                if tensor.dtype not in SUPPORTED_DTYPES:
                    print(f"Skipping unsupported tensor: {key}, dtype: {tensor.dtype}")
                    tensors_skipped += 1
                    continue

                # Quantize tensor if float32 or bfloat16
                if tensor.dtype == torch.float32 or tensor.dtype == torch.bfloat16:
                    print(f"Quantizing tensor: {key}, dtype: {tensor.dtype}, shape: {tensor.shape}")
                    scale = (tensor.max() - tensor.min()) / 255.0 if tensor.numel() > 0 else 1.0
                    zero_point = int(-tensor.min() / scale) if tensor.numel() > 0 else 0
                    quantized_tensors[key] = torch.quantize_per_tensor(
                        tensor.float(), scale=scale, zero_point=zero_point, dtype=torch.qint8
                    ).dequantize().to(torch.int8)
                elif tensor.dtype in SUPPORTED_DTYPES:
                    quantized_tensors[key] = tensor
                else:
                    print(f"Skipping unsupported tensor: {key}, dtype: {tensor.dtype}")

        shards_processed += 1

    except Exception as e:
        print(f"Error processing shard {shard_file}: {e}")

# Check if any shards were successfully processed
if shards_processed == 0:
    print("No valid shards processed. Exiting.")
    exit(1)

# Validate tensors before saving
for key, tensor in quantized_tensors.items():
    print(f"Validating Tensor: {key}")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Num elements: {tensor.numel()}")
    print(f"  Min: {tensor.min().item() if tensor.numel() > 0 else 'N/A'}")
    print(f"  Max: {tensor.max().item() if tensor.numel() > 0 else 'N/A'}")

    if tensor.numel() > 0:
        if tensor.dtype in {torch.float32, torch.bfloat16, torch.float64}:
            mean_value = tensor.mean().item()
        else:
            mean_value = tensor.float().mean().item()  # Cast to float for mean calculation
        print(f"  Mean: {mean_value}")
    else:
        print("  Mean: N/A")
    print("-" * 50)


# Save all quantized tensors into a single .safetensors file
output_file = os.path.join(output_dir, "model.safetensors")
try:
    if quantized_tensors:
        save_file(quantized_tensors, output_file)
        print(f"Quantized model saved to {output_file}")
    else:
        print("No tensors were quantized. Skipping saving.")
except Exception as e:
    print(f"Error saving quantized model: {e}")
    exit(1)

# Copy config.json with validation
config_path = os.path.join(local_model_path, "config.json")
output_config_path = os.path.join(output_dir, "config.json")
try:
    if os.path.exists(config_path):
        os.system(f"cp {config_path} {output_config_path}")
        print(f"Config copied to {output_config_path}")
    else:
        print("No config.json found. Ensure you provide one.")
except Exception as e:
    print(f"Error copying config file: {e}")

# Summary of results
print(f"Quantization completed. Shards processed: {shards_processed}, tensors skipped: {tensors_skipped}.")
