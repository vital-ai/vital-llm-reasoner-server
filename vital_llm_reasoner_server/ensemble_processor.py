import logging
from vital_llm_reasoner_server.ensemble_worker import get_ensemble_worker

BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"

# inject ensemble result
search_result = "Jimmy Carter's birthday is: October 1, 1924"

class EnsembleLogitsProcessor:

    def __init__(self):
        self.gen_buffer = ""
        self.result_count = 0

    def __call__(self, prompt_tokens_ids, past_tokens_ids, logits_row):

        ensemble_worker = get_ensemble_worker()

        tokenizer = ensemble_worker.get_tokenizer()

        if past_tokens_ids:

            prior_token_id = past_tokens_ids[-1]

            logging.info(f"Prior token ID: {prior_token_id}")

            prior_token = tokenizer.decode(prior_token_id)

            logging.info(f"Prior token: {prior_token}")

            self.gen_buffer += prior_token

            if END_SEARCH_QUERY in self.gen_buffer:

                encoded_result = f"{BEGIN_SEARCH_RESULT}{search_result}{END_SEARCH_RESULT}"

                tokens = tokenizer.encode(encoded_result)

                if self.result_count < len(tokens):

                    logits_row[:] = logits_row[:] = -65504 # -float('inf')  # Mask all tokens

                    token_id = tokens[self.result_count]  # self.tokenizer.encode(tokens[self.result_count])[0]

                    logits_row[token_id] = +65504  # float('inf')

                    self.result_count += 1

                    # modified scores
                    return logits_row

        else:
            logging.info("No prior token (beginning of generation).")

        return logits_row
