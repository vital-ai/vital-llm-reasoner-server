# API endpoint and headers
import json
import time
import requests

"""
--host 0.0.0.0 --port 8000 --model Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ --enforce-eager --gpu-memory-utilization 0.95 --api-key sk-deepseek-testing --max-model-len 8128 --quantization awq_marlin
"""

# Valdemardi/DeepSeek-R1-Distill-Llama-70B-AWQ

url = "https://r9p0430vxvtqp5-8000.proxy.runpod.net/v1/completions"

# url = "http://localhost:11435/v1/completions"

headers = {
    "Authorization": "Bearer sk-deepseek-testing",
    "Content-Type": "application/json",
}

MAX_SEARCH_LIMIT = 5

# saying "be concise" reduces tokens almost by 2/3 (2020 to 790)
# not sure if it affects quality

user_text_1 = """
Solve this puzzle and be concise in your reasoning.

Selena, Jennifer and Miley wear a blue dress, yellow dress, and green dress in an unknown order. It is known that:

1) If Selena wears blue, then Jennifer wears green.
2) If Selena wears yellow, then Miley wears green.
3) If Jennifer does not wear yellow, then Miley wears blue.

What is the color of the dress Selena is wearing?
”"""

user_text_2 = """What is Jimmy Carter's birthday?"""

instruction_1 = "You are a helpful reasoning assistant."

instruction_2 = f"""You are a reasoning assistant with the ability to use tools to perform web searches to help
you answer the user's question accurately.
For factual questions, you must use these tools.  Do not trust your own memory.
- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.
Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.
You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.
Once you have all the information you need, continue your reasoning and answer the question.
Example:
Question: Who got the first Nobel Prize in Physics?
Assistant thinking steps:
- I need to find out who was awarded the first Nobel Prize in Physics.
Assistant:
<|begin_search_query|>first Nobel Prize in Physics winner<|end_search_query|>
(System returns processed information from relevant web pages)
<|begin_search_result|>Wilhelm Conrad Röntgen won the first Nobel Prize in Physics in 1901 for discovering X-rays<|end_search_result|>
Assistant continues reasoning with the new information...
Remember:
- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.
- Only trust the information in: <|begin_search_result|>Search Results Here<|end_search_result|> for confirmation of factual information.
- When done searching, continue your reasoning until you determine an answer.
- You must provide your final answer in the format \\boxed{{YOUR_ANSWER}}
"""


# Prompt components in ChatML format
system_message = f"""
<system
{instruction_2}

"""

user_message = f"""user
{user_text_1}
"""

assistant_message = "assistant\n"

# Combine the components into the final prompt
prompt = f"{system_message}{user_message}{assistant_message}"

# Request payload
payload = {
    # "model": "mlx-community/DeepSeek-R1-Distill-Llama-70B-4bit", # "Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ",
    "model": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-MLX-4Bit",
    "prompt": prompt,
    "max_tokens": 2500,
    "temperature": 0.7,
    "stream": True,
}

start_time = time.time()
total_tokens = 0  # Track total tokens generated


# Send the request with streaming enabled
response = requests.post(url, headers=headers, json=payload, stream=True)

# Check for response status
if response.status_code == 200:
    print("Streaming response text:")
    output_text = ""

    try:
        for line in response.iter_lines(decode_unicode=True):
            if line == "data: [DONE]":
                print("\n[Stream completed]")
                break
            if line.startswith("data: "):  # Process lines starting with "data: "
                try:
                    # Parse the JSON content after "data: "
                    data = json.loads(line[len("data: "):])
                    # Extract the text field
                    text = data.get("choices", [{}])[0].get("text", "")
                    output_text += text  # Append to the output
                    total_tokens += len(text.split())  # Estimate tokens by splitting on spaces

                    print(text, end="", flush=True)  # Print text as it streams
                except json.JSONDecodeError:
                    print("\nError decoding JSON:", line)
    except Exception as e:
        print(f"\nError during streaming: {e}")
else:
    print(f"Request failed with status code {response.status_code}: {response.text}")

# Calculate and display elapsed time
elapsed_time = time.time() - start_time
tps = total_tokens / elapsed_time if elapsed_time > 0 else 0  # Tokens per second

print(f"\n\nRequest completed in {elapsed_time:.2f} seconds.")
print(f"Total tokens generated: {total_tokens}")
print(f"Tokens per second (TPS): {tps:.2f}")
