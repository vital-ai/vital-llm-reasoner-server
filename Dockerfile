FROM vllm/vllm-openai:latest

WORKDIR /app

# pip install this
COPY vital_llm_reasoner_server/ /app/vital_llm_reasoner_server/
COPY setup.py /app/setup.py

COPY logging_config.json /app/logging_config.json

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# important to install to allow spawned processes to find the code
RUN pip install --no-cache-dir -e .

EXPOSE 8000

ENV VLLM_LOGGING_CONFIG_PATH="/app/logging_config.json"
ENV VLLM_LOGGING_LEVEL="INFO"

# this should use the files in /app
ENTRYPOINT ["python3", "vital_llm_reasoner_server/ensemble_server.py"]

# deploying in runpod overrides this command if desired
# which allows setting key and other parameters
# quantization of this model could use "awq_marlin" or "awq"
# marlin should be faster but potentially less stable implementation

CMD [ "--host", "0.0.0.0", "--port", "8000", \
     "--api-key", "sk-qwq-testing", \
     "--enforce-eager", \
     "--model", "KirillR/QwQ-32B-Preview-AWQ", \
     "--gpu-memory-utilization", "0.95", \
     "--max-model-len", "8128", "--quantization", "awq_marlin"]
