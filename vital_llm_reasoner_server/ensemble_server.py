import logging
from typing import Optional, Callable, Awaitable
import uvloop
from fastapi.routing import APIRoute
from fastapi import APIRouter, FastAPI, HTTPException, Request

from pydantic import Field
from vllm import LLMEngine, AsyncEngineArgs, SamplingParams
from vllm.entrypoints.openai.api_server import run_server, router
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.openai.protocol import get_logits_processors, CompletionRequest, ChatCompletionRequest
from vllm.sampling_params import GuidedDecodingParams, RequestOutputKind
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.transformers_utils.tokenizer_group import BaseTokenizerGroup
from vllm.utils import FlexibleArgumentParser
from vital_llm_reasoner_server.ensemble_manager import get_ensemble_manager
from vital_llm_reasoner_server.ensemble_processor import EnsembleLogitsProcessor
from vital_llm_reasoner_server.ensemble_worker import get_ensemble_worker

# currently this only received the prompt and current generation
# without more details about the request, like the user, etc.
# current plan to encode that info in the prompt as part of the system prompt
# may include a request id in the prompt to simplify looking up info

# current assumption is that a generation will occur in the same backend process
# but if not the info in the prompt and current generation should be enough to re-build
# state

# additional JWT field passed in to use with kgraphservice and tool requests
class VitalCompletionRequest(CompletionRequest):
    jwt_auth: Optional[str] = Field(default=None, description="JWT authorization")

OriginalFunctionType = Callable[[CompletionRequest, Request], Awaitable[dict]]

original_create_completion: Optional[OriginalFunctionType] = None

async def vital_create_completion(request: VitalCompletionRequest, raw_request: Request):

    user = None

    if request.user:
        user = request.user
    else:
        # generate something?
        pass

    if request.jwt_auth:
        print(f"User: {user} : JWT Authorization: {request.jwt_auth}")
        # assert user --> jwt which can be used downstream by the orchestrator

    if original_create_completion:
        return await original_create_completion(request, raw_request)

def to_sampling_params(
        self,
        default_max_tokens: int,
        logits_processor_pattern: Optional[str],
        default_sampling_params: Optional[dict] = None) -> SamplingParams:

    max_tokens = self.max_tokens

    if max_tokens is None:
        max_tokens = default_max_tokens

    logging.info('patched to_sample_params called')

    ensemble_manager = get_ensemble_manager()

    # this should be non-abstract class with a tokenizer instance
    # tokenizer_group = get_tokenizer_group()
    # tokenizer: AnyTokenizer = tokenizer_group.tokenizer if tokenizer_group else None

    if default_sampling_params is None:
        default_sampling_params = {}

    # Default parameters
    if (repetition_penalty := self.repetition_penalty) is None:
        repetition_penalty = default_sampling_params.get(
            "repetition_penalty",
            self._DEFAULT_SAMPLING_PARAMS["repetition_penalty"],
        )
    if (temperature := self.temperature) is None:
        temperature = default_sampling_params.get(
            "temperature", self._DEFAULT_SAMPLING_PARAMS["temperature"])
    if (top_p := self.top_p) is None:
        top_p = default_sampling_params.get(
            "top_p", self._DEFAULT_SAMPLING_PARAMS["top_p"])
    if (top_k := self.top_k) is None:
        top_k = default_sampling_params.get(
            "top_k", self._DEFAULT_SAMPLING_PARAMS["top_k"])
    if (min_p := self.min_p) is None:
        min_p = default_sampling_params.get(
            "min_p", self._DEFAULT_SAMPLING_PARAMS["min_p"])

    prompt_logprobs = self.prompt_logprobs
    if prompt_logprobs is None and self.echo:
        prompt_logprobs = self.logprobs

    echo_without_generation = self.echo and self.max_tokens == 0

    guided_json_object = None
    if (self.response_format is not None
            and self.response_format.type == "json_object"):
        guided_json_object = True

    guided_decoding = GuidedDecodingParams.from_optional(
        json=self.guided_json,
        regex=self.guided_regex,
        choice=self.guided_choice,
        grammar=self.guided_grammar,
        json_object=guided_json_object,
        backend=self.guided_decoding_backend,
        whitespace_pattern=self.guided_whitespace_pattern)

    logits_processors = get_logits_processors(self.logits_processors,
                                              logits_processor_pattern)

    # ensemble_logits_processor = {"qualname": "vital_llm_reasoner_server.ensemble_server.LoggingLogitsProcessor"}
    ensemble_logits_processor = EnsembleLogitsProcessor()

    if logits_processors:
        logits_processors.append(ensemble_logits_processor)
    else:
        logits_processors = [ensemble_logits_processor]

    return SamplingParams.from_optional(
        n=self.n,
        best_of=self.best_of,
        presence_penalty=self.presence_penalty,
        frequency_penalty=self.frequency_penalty,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        seed=self.seed,
        stop=self.stop,
        stop_token_ids=self.stop_token_ids,
        logprobs=self.logprobs,
        ignore_eos=self.ignore_eos,
        max_tokens=max_tokens if not echo_without_generation else 1,
        min_tokens=self.min_tokens,
        prompt_logprobs=prompt_logprobs,
        skip_special_tokens=self.skip_special_tokens,
        spaces_between_special_tokens=self.spaces_between_special_tokens,
        include_stop_str_in_output=self.include_stop_str_in_output,
        logits_processors=logits_processors,
        truncate_prompt_tokens=self.truncate_prompt_tokens,
        output_kind=RequestOutputKind.DELTA if self.stream \
            else RequestOutputKind.FINAL_ONLY,
        guided_decoding=guided_decoding,
        logit_bias=self.logit_bias,
        allowed_token_ids=self.allowed_token_ids)


# this won't work if the LLMEngine is in the back end process
# original_init = LLMEngine.__init__
# def patched_init(self, *args, **kwargs):
#    original_init(self, *args, **kwargs)
#    engine_list.append(self)
# LLMEngine.__init__ = patched_init

# TODO patch openai requests to include optional JWT parameter?
# JWT would be passed to the ensemble orchestrator to use with the ensemble member requests

def patch_create_completion():

    global original_create_completion

    for route in router.routes:
        if isinstance(route, APIRoute) and route.path == "/v1/completions" and route.methods == {"POST"}:

            original_create_completion = route.endpoint
            route.endpoint = vital_create_completion
            route.dependant.dependencies[0].model = VitalCompletionRequest
            break

def patch_vllm():

    CompletionRequest.to_sampling_params = to_sampling_params
    patch_create_completion()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    patch_vllm()

    # logging.getLogger("vllm").setLevel(logging.DEBUG)
    # logging.getLogger("huggingface_hub").setLevel(logging.DEBUG)

    logging.info("Starting Vital.ai Patched vLLM server...")
    logging.info("Version 0.0.9")

    parser = FlexibleArgumentParser(
        description="Vital.ai Patched vLLM OpenAI-Compatible REST API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    model = args.model

    global_model = model

    uvloop.run(run_server(args))
