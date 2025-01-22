import logging
from threading import Lock

from transformers import AutoTokenizer

class EnsembleManager:
    _instance = None  # For enforcing a single instance
    _shared_state = {}  # Shared state across instances (Borg pattern)
    _lock = Lock()


    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # Double-checked locking
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Ensure this is only initialized once
        if not hasattr(self, "_initialized"):
            self._initialized = True

            # hard code for now
            self._shared_state["model"] = "KirillR/QwQ-32B-Preview-AWQ"

            logging.info(f"EnsembleManager initialized.")

    def set_model_name(self, model):
        self._shared_state["model"] = model

    def get_model_name(self):
        return self._shared_state.get("model", None)

    def get_tokenizer(self):

        with self._lock:  # Ensure only one thread initializes the tokenizer
            tokenizer = self._shared_state.get("tokenizer", None)
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.get_model_name(),
                    trust_remote_code=True
                )
                self._shared_state["tokenizer"] = tokenizer
                logging.info("Tokenizer initialized.")
            return tokenizer


# Singleton accessor
def get_ensemble_manager():
    return EnsembleManager()

# access ensemble resources

# ensemble conductor and members act in the backend processes

# potentially the server side would need to inject info into
# the prompt such as what ensemble members are present

