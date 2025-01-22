from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# import bitsandbytes as bnb

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

model_name = "QwQ-32B-Preview"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Automatically map layers to available GPUs

    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save quantized model
model.save_pretrained("qwq_quantized")
tokenizer.save_pretrained("qwq_quantized")

print("Model quantized and saved to 'qwq_quantized'")
