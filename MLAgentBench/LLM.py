""" This file contains the code for calling all LLM APIs. """

import os
from functools import partial
import tiktoken
from .schema import TooLongPromptError, LLMError
import load_dotenv
load_dotenv.load_dotenv()

enc = tiktoken.get_encoding("cl100k_base")

try:
    from helm.common.authentication import Authentication
    from helm.common.request import Request, RequestResult
    from helm.proxy.accounts import Account
    from helm.proxy.services.remote_service import RemoteService
    # setup CRFM API
    auth = Authentication(api_key=open("crfm_api_key.txt").read().strip())
    service = RemoteService("https://crfm-models.stanford.edu")
    account: Account = service.get_account(auth)
except Exception as e:
    print(e)
    print("Could not load CRFM API key crfm_api_key.txt.")

try:   
    import anthropic
    # setup anthropic API key
    anthropic_client = anthropic.Anthropic(api_key=open("claude_api_key.txt").read().strip())
except Exception as e:
    anthropic = None
    print(e)
    print("Could not load anthropic API key claude_api_key.txt.")

HUMAN_PROMPT = "\n\nHuman:"
AI_PROMPT = "\n\nAssistant:"
    
try:
    import openai
    # setup OpenAI API key
    openai.organization, openai.api_key  =  open("openai_api_key.txt").read().strip().split(":")    
    os.environ["OPENAI_API_KEY"] = openai.api_key 
except Exception as e:
    print(e)
    print("Could not load OpenAI API key openai_api_key.txt.")

try:
    import vertexai
    from vertexai.preview.generative_models import GenerativeModel, Part
    from google.cloud.aiplatform_v1beta1.types import SafetySetting, HarmCategory
    vertexai.init(project=PROJECT_ID, location="us-central1")
except Exception as e:
    print(e)
    print("Could not load VertexAI API.")

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
import torch

loaded_hf_models = {}

class StopAtSpecificTokenCriteria(StoppingCriteria):
    def __init__(self, stop_sequence):
        super().__init__()
        self.stop_sequence = stop_sequence

    def __call__(self, input_ids, scores, **kwargs):
        # Create a tensor from the stop_sequence
        stop_sequence_tensor = torch.tensor(self.stop_sequence, device=input_ids.device, dtype=input_ids.dtype)

        # Check if the current sequence ends with the stop_sequence
        current_sequence = input_ids[:, -len(self.stop_sequence) :]
        return bool(torch.all(current_sequence == stop_sequence_tensor).item())

    
def log_to_file(log_file, prompt, completion, model, max_tokens_to_sample, api_metadata=None):
    """ Log the prompt and completion to a file."""
    import json
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n===================prompt=====================\n")
        f.write(f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}")
        num_prompt_tokens = len(enc.encode(f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}"))
        f.write(f"\n==================={model} response ({max_tokens_to_sample})=====================\n")
        f.write(completion)
        num_sample_tokens = len(enc.encode(completion))
        f.write("\n===================tokens=====================\n")
        f.write(f"Estimated prompt tokens (local tokenizer): {num_prompt_tokens}\n")
        f.write(f"Estimated visible completion tokens (local tokenizer): {num_sample_tokens}\n")
        if api_metadata is not None:
            f.write("\n===================API usage and response metadata=====================\n")
            f.write(json.dumps(api_metadata, ensure_ascii=False, indent=2) + "\n")
            f.write("\n===================reasoning summaries (not raw reasoning)=====================\n")
            f.write("\n\n".join(api_metadata.get("reasoning_summaries", [])) or "No reasoning summary returned.")
        f.write("\n\n")
    if api_metadata is not None:
        with open(str(log_file) + ".usage.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(api_metadata, ensure_ascii=False) + "\n")


def request_openai_responses(payload, timeout=600):
    """Use Responses without upgrading this project's legacy OpenAI SDK."""
    import requests
    key = getattr(openai, "api_key", None) or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise LLMError("An OpenAI API key is required.")
    base = os.environ.get("OPENAI_BASE_URL") or getattr(openai, "api_base", None) or "https://api.openai.com/v1"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    organization = getattr(openai, "organization", None) or os.environ.get("OPENAI_ORG_ID")
    if organization:
        headers["OpenAI-Organization"] = organization
    if os.environ.get("OPENAI_PROJECT_ID"):
        headers["OpenAI-Project"] = os.environ["OPENAI_PROJECT_ID"]
    response = requests.post(base.rstrip("/") + "/responses", json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()

def complete_text_hf(prompt, stop_sequences=[], model="huggingface/codellama/CodeLlama-7b-hf", max_tokens_to_sample = 2000, temperature=0.5, log_file=None, **kwargs):
    model = model.split("/", 1)[1]
    if model in loaded_hf_models:
        hf_model, tokenizer = loaded_hf_models[model]
    else:
        tokenizer = AutoTokenizer.from_pretrained(model)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        hf_model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=dtype).to(device)
        hf_model.eval()
        loaded_hf_models[model] = (hf_model, tokenizer)

    if "qwen3" in model.lower():
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded_input = tokenizer(text, return_tensors="pt", return_token_type_ids=False).to(hf_model.device)
    else:
        encoded_input = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(hf_model.device)
    stop_sequence_ids = tokenizer(stop_sequences, return_token_type_ids=False, add_special_tokens=False)
    stopping_criteria = StoppingCriteriaList()
    for stop_sequence_input_ids in stop_sequence_ids.input_ids:
        stopping_criteria.append(StopAtSpecificTokenCriteria(stop_sequence=stop_sequence_input_ids))

    output = hf_model.generate(
        **encoded_input,
        temperature=temperature,
        max_new_tokens=max_tokens_to_sample,
        do_sample=True,
        return_dict_in_generate=True,
        output_scores=True,
        stopping_criteria = stopping_criteria,
        **kwargs,
    )
    sequences = output.sequences
    sequences = [sequence[len(encoded_input.input_ids[0]) :] for sequence in sequences]
    all_decoded_text = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    completion = all_decoded_text[0]
    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
    return completion

def complete_text_gemini(prompt, stop_sequences=[], model="gemini-pro", max_tokens_to_sample = 2000, temperature=0.5, log_file=None, **kwargs):
    """ Call the gemini API to complete a prompt."""
    # Load the model
    model = GenerativeModel("gemini-pro")
    # Query the model
    parameters = {
            "temperature": temperature,
            "max_output_tokens": max_tokens_to_sample,
            "stop_sequences": stop_sequences,
            **kwargs
        }
    safety_settings = {
            harm_category: SafetySetting.HarmBlockThreshold(SafetySetting.HarmBlockThreshold.BLOCK_NONE)
            for harm_category in iter(HarmCategory)
        }
    safety_settings = {
        }
    response = model.generate_content( [prompt], generation_config=parameters, safety_settings=safety_settings)
    completion = response.text
    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
    return completion

def complete_text_claude(prompt, stop_sequences=[HUMAN_PROMPT], model="claude-v1", max_tokens_to_sample = 2000, temperature=0.5, log_file=None, messages=None, **kwargs):
    """ Call the Claude API to complete a prompt."""

    if anthropic is None:
        raise LLMError("Anthropic is unavailable. Use a huggingface/... model or install and configure Anthropic.")
    ai_prompt = AI_PROMPT
    if "ai_prompt" in kwargs is not None:
        ai_prompt = kwargs["ai_prompt"]

    
    try:
        if model == "claude-3-opus-20240229":
            while True:
                try:
                    message = anthropic_client.messages.create(
                        messages=[
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ] if messages is None else messages,
                        model=model,
                        stop_sequences=stop_sequences,
                        temperature=temperature,
                        max_tokens=max_tokens_to_sample,
                        **kwargs
                    )
                except anthropic.InternalServerError as e:
                    pass
                try:
                    completion = message.content[0].text
                    break
                except:
                    print("end_turn???")
                    pass
        else:
            rsp = anthropic_client.completions.create(
                prompt=f"{HUMAN_PROMPT} {prompt} {ai_prompt}",
                stop_sequences=stop_sequences,
                model=model,
                temperature=temperature,
                max_tokens_to_sample=max_tokens_to_sample,
                **kwargs
            )
            completion = rsp.completion
        
    except anthropic.APIStatusError as e:
        print(e)
        raise TooLongPromptError()
    except Exception as e:
        raise LLMError(e)

    
    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
    return completion


def get_embedding_crfm(text, model="openai/gpt-4-0314"):
    request = Request(model="openai/text-embedding-ada-002", prompt=text, embedding=True)
    request_result: RequestResult = service.make_request(auth, request)
    return request_result.embedding 
    
def complete_text_crfm(prompt="", stop_sequences = [], model="openai/gpt-4-0314",  max_tokens_to_sample=2000, temperature = 0.5, log_file=None, messages = None, **kwargs):
    
    random = log_file
    if messages:
        request = Request(
                prompt=prompt, 
                messages=messages,
                model=model, 
                stop_sequences=stop_sequences,
                temperature = temperature,
                max_tokens = max_tokens_to_sample,
                random = random
            )
    else:
        # print("model", model)
        # print("max_tokens", max_tokens_to_sample)
        request = Request(
                # model_deployment=model,
                prompt=prompt, 
                model=model, 
                stop_sequences=stop_sequences,
                temperature = temperature,
                max_tokens = max_tokens_to_sample,
                random = random
        )
    
    try:      
        request_result: RequestResult = service.make_request(auth, request)
    except Exception as e:
        # probably too long prompt
        print(e)
        raise TooLongPromptError()
    
    if request_result.success == False:
        print(request.error)
        raise LLMError(request.error)
    completion = request_result.completions[0].text
    if log_file is not None:
        log_to_file(log_file, prompt if not messages else str(messages), completion, model, max_tokens_to_sample)
    return completion


def complete_text_openai(prompt, stop_sequences=None, model="gpt-3.5-turbo", max_tokens_to_sample=8000, temperature=0.2, log_file=None, **kwargs):
    """Use Responses for reasoning models; retain existing chat/text routes."""
    import re
    base_model = model.split(":")[1] if model.startswith("ft:") else model
    reasoning = bool(re.match(r"(?:gpt-(?:[5-9]|[1-9][0-9])(?:[.\-]|$)|o[1-9](?:-|$))", base_model))
    chat = (base_model.startswith(("gpt-", "chatgpt-")) or reasoning) and "instruct" not in base_model
    raw_request = {"model": model, **kwargs}
    stops = raw_request.pop("stop", stop_sequences) or []
    stops = [stops] if isinstance(stops, str) else stops
    summaries = []
    if reasoning:
        token_limit = raw_request.pop("max_tokens", max_tokens_to_sample)
        token_limit = raw_request.pop("max_completion_tokens", token_limit)
        raw_request.setdefault("max_output_tokens", token_limit)
        config = dict(raw_request.pop("reasoning", None) or {})
        effort = raw_request.pop("reasoning_effort", None)
        if effort is not None:
            config.setdefault("effort", effort)
        config["summary"] = "auto"
        raw_request["reasoning"] = config
        raw_request.setdefault("store", False)
        for key in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
            raw_request.pop(key, None)
        timeout = raw_request.pop("request_timeout", 600)
        response = request_openai_responses({"input": prompt, **raw_request}, timeout=timeout)
        text_parts = []
        for item in response.get("output", []):
            if item.get("type") == "message" and item.get("role") == "assistant":
                text_parts.extend(part["text"] for part in item.get("content", [])
                                  if part.get("type") == "output_text")
            elif item.get("type") == "reasoning":
                summaries.extend(part["text"] for part in item.get("summary", [])
                                 if part.get("type") == "summary_text")
        completion = "".join(text_parts)
        finish_reason = response.get("incomplete_details") or response.get("status")
        endpoint = "responses"
    else:
        raw_request.setdefault("temperature", temperature)
        raw_request.setdefault("max_tokens", max_tokens_to_sample)
        raw_request["stop"] = stops or None
        if chat:
            response = openai.ChatCompletion.create(messages=[{"role": "user", "content": prompt}], **raw_request)
            choice = response["choices"][0]
            completion = choice["message"]["content"]
            endpoint = "chat/completions"
        else:
            response = openai.Completion.create(prompt=prompt, **raw_request)
            choice = response["choices"][0]
            completion = choice["text"]
            endpoint = "completions"
        finish_reason = choice.get("finish_reason")
    metadata = {
        "endpoint": endpoint, "response_id": response.get("id"),
        "model": response.get("model", model), "usage": response.get("usage"),
        "finish_reason": finish_reason, "reasoning_summaries": summaries,
    }
    # Log full returned text and usage even for an exhausted reasoning budget.
    # Usage includes all generated tokens, including locally truncated text.
    if log_file is not None:
        log_to_file(log_file, prompt, completion or "", model, max_tokens_to_sample,
                    api_metadata=metadata)
    if not isinstance(completion, str) or not completion.strip():
        raise LLMError(f"{model} returned no text (finish_reason={finish_reason}). "
                       "For reasoning models, the completion-token budget includes reasoning tokens; "
                       "increase that budget if finish_reason is length.")
    if reasoning:
        positions = [completion.find(stop) for stop in stops if stop and stop in completion]
        if positions:
            completion = completion[:min(positions)]
    return completion

def complete_text(prompt, log_file, model, **kwargs):
    """ Complete text using the specified model with appropriate API. """
    
    if model.startswith("claude"):
        # use anthropic API
        completion = complete_text_claude(prompt, stop_sequences=[HUMAN_PROMPT, "Observation:"], log_file=log_file, model=model, **kwargs)
    elif model.startswith("gemini"):
        completion = complete_text_gemini(prompt, stop_sequences=["Observation:"], log_file=log_file, model=model, **kwargs)
    elif model.startswith("huggingface"):
        completion = complete_text_hf(prompt, stop_sequences=["Observation:"], log_file=log_file, model=model, **kwargs)
    elif "/" in model:
        # use CRFM API since this specifies organization like "openai/..."
        completion = complete_text_crfm(prompt, stop_sequences=["Observation:"], log_file=log_file, model=model, **kwargs)
    else:
        # use OpenAI API
        completion = complete_text_openai(prompt, stop_sequences=["Observation:"], log_file=log_file, model=model, **kwargs)
    return completion

# specify fast models for summarization etc
FAST_MODEL = "claude-v1"
def complete_text_fast(prompt, **kwargs):
    return complete_text(prompt = prompt, model = FAST_MODEL, temperature =0.01, **kwargs)
# complete_text_fast = partial(complete_text_openai, temperature= 0.01)

