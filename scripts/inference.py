from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# model_name = "qwq_quantized"

model_name = "QwQ-32B-Preview"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.model_max_length = 8000


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# input_text = "What is the capital of France?"

input_text = """
“Selena, Jennifer and Miley wear a blue dress, yellow dress, and green dress in 
an unknown order. It is known that:

1) If Selena wears blue, then Jennifer wears green.
2) If Selena wears yellow, then Miley wears green.
3) If Jennifer does not wear yellow, then Miley wears blue.

What is the color of the dress Selena is wearing?”
"""


# inputs = tokenizer(input_text, return_tensors="pt").to(device)


inputs = tokenizer(input_text, return_tensors="pt").to(device)

input_ids = inputs["input_ids"].to(torch.long)

attention_mask = inputs["attention_mask"]

max_new_tokens = 2000

for _ in range(max_new_tokens):
    # Get model output
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    # Select the next token (argmax or sampling)
    next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)

    # Append the token to the input
    input_ids = torch.cat([input_ids, next_token_id], dim=-1)
    attention_mask = torch.cat(
        [attention_mask, torch.ones((attention_mask.shape[0], 1), device=device)
], dim=-1
    )

    # Decode the new token and print
    decoded_token = tokenizer.decode(next_token_id.squeeze(), skip_special_tokens=True)
    print(decoded_token, end="", flush=True)

    # Stop if the model generates an end-of-sequence token
    if next_token_id.squeeze().item() == tokenizer.eos_token_id:
        break


"""
outputs = model.generate(
        **inputs,
        max_length=2000,  # Set maximum combined length of input and output toke
ns
        max_new_tokens=2000  # Limit the maximum number of tokens generated
        )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
"""

