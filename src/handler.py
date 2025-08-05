# Native
import os
import time
import traceback
import logging
import re
import math
import json
from pathlib import Path
from types import SimpleNamespace

# Files
import llm_engine_patch

# Dependencies
import runpod
from dotenv import load_dotenv
from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine
from vllm.usage.usage_lib import UsageContext

from vllm.entrypoints.openai.protocol import ChatCompletionRequest, CompletionRequest, EmbeddingChatRequest, ErrorResponse
from vllm.entrypoints.openai.serving_models import OpenAIServingModels, BaseModelPath, LoRAModulePath
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding

# ---------------------------------------------------------------------------- #
#                             Initialization                                   #
# ---------------------------------------------------------------------------- #
load_dotenv() # Load .env file
vllm_concurrency = 1 # Default

engine = None
engine_args = None
api_engine_config = None
api_served_models = None
tokenizer = None

async def initialize_engine(input):
    global engine_args
    global engine
    global vllm_concurrency
    global api_engine_config
    global api_served_models
  
    if not engine:
        print("VLLM Engine not initialized, starting...")
        st = time.time()

        # Get .env config, combine it with dynamic request values and prepare it for the engine
        engine_args = {**get_args({}, "VLLME_"), **input["openai_input"].get("vllm_config", {}).get("engine_args", {})}
        prefixes = ("VLLM_", "OPENAI_", "VLLMC_", "TORCHDYNAMO_")
        for key, value in input["openai_input"].get("vllm_config", {}).get("env", {}).items():
            if key.startswith(prefixes):
                os.environ[key] = str(value)

        # Auto network-volume torch.compile
        auto_network_cache()

        # Method to capture vLLM concurrency log so we can dynamically set RunPod concurrency based on current GPU capacity
        logger = logging.getLogger('vllm')
        concurrency_handler = ConcurrencyLogHandler()
        logger.setLevel(logging.INFO)
        # Init engine
        logger.addHandler(concurrency_handler)
        engine = AsyncLLMEngine.from_engine_args(engine_args=AsyncEngineArgs(**engine_args), usage_context=UsageContext.OPENAI_API_SERVER)
        logger.removeHandler(concurrency_handler)
        # Finish and process concurrency capture
        vllm_concurrency = max(1, math.floor(float(concurrency_handler.concurrency)))
        print(f"-- Captured and set worker's concurrency capability: {vllm_concurrency}")

        initialize_tokenizer()

        # Load LoRA adapters
        api_engine_config = await engine.get_model_config()
        lora_modules = os.getenv('LORA_ADAPTERS', None)
        if lora_modules is not None:
            try:
                lora_modules = json.loads(lora_modules)
                lora_modules = [LoRAModulePath(**lora_modules)]
            except: lora_modules = None

        api_served_models = OpenAIServingModels(engine_client=engine, model_config=api_engine_config, base_model_paths=[BaseModelPath(name=engine_args["model"], model_path=engine_args["model"])], lora_modules=lora_modules, prompt_adapters=None)
        print('─' * 20, "--- VLLM engine and model cold start initialization took %s seconds ---" % (time.time() - st), '─' * 20, "*Unfortunately, this time is being billed.", sep=os.linesep)

def initialize_tokenizer():
    global tokenizer

    if not tokenizer:
        print("Tokenizer not initialized, starting...")
        pretrained_model_name_or_path = engine_args.get("tokenizer") or engine_args.get("model")
        revision = engine_args.get("tokenizer_revision", "main")
        trust_remote_code = engine_args.get("trust_remote_code", False)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path, revision=revision, trust_remote_code=trust_remote_code)
        if os.getenv("VLLMC_CUSTOM_CHAT_TEMPLATE",None):
            tokenizer.chat_template = os.getenv("VLLMC_CUSTOM_CHAT_TEMPLATE",None)

# ---------------------------------------------------------------------------- #
#                            Helper Functions                                  #
# ---------------------------------------------------------------------------- #
class ConcurrencyLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.concurrency = 1
    def emit(self, record):
        log_message = self.format(record)
        match = re.search(r".*?Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x", log_message)
        if match:
            self.concurrency = float(match.group(1))

def concurrency_modifier(current_concurrency):
    current_concurrency = vllm_concurrency - current_concurrency 
    return current_concurrency

class BatchSize:
    def __init__(self, max_batch_size, min_batch_size, batch_size_growth_factor):
        self.max_batch_size = max_batch_size
        self.batch_size_growth_factor = batch_size_growth_factor
        self.min_batch_size = min_batch_size
        self.is_dynamic = batch_size_growth_factor > 1 and min_batch_size >= 1 and max_batch_size > min_batch_size
        if self.is_dynamic:
            self.current_batch_size = min_batch_size
        else:
            self.current_batch_size = max_batch_size
    def update(self):
        if self.is_dynamic:
            self.current_batch_size = min(self.current_batch_size*self.batch_size_growth_factor, self.max_batch_size)

def get_args(obj, prefix):
    def is_number(v):
        try:
            num = float(v)
            return int(num) if num.is_integer() else num
        except ValueError: return False
    def convert_value(v):
        if isinstance(v, bool) or isinstance(v, (int, float)): return v
        if v.strip().lower() in ['none', '']: return None
        if v.lower() in ['true', 'false']: return v.lower() == 'true'
        if v.startswith('{') or v.startswith('['):
            try: return json.loads(v)
            except json.JSONDecodeError: pass
        num = is_number(v)
        if num is not False: v = num
        return v
    defaults = {}
    for key, value in os.environ.items():
        if key.startswith(prefix) and (value.upper() != "NULL" and value is not None): defaults[key.replace(prefix,"").lower()] = convert_value(value)
    for key, value in obj.items(): convert_value(value)
    res = {**defaults, **obj}
    return res

def auto_network_cache():
    if os.getenv("VLLMC_AUTO_NETWORK_CACHE") == "true":
        val = "0" if Path("/runpod-volume").exists() else "1"
        os.environ["VLLM_DISABLE_COMPILE_CACHE"] = val
        os.environ["TORCHDYNAMO_DISABLE"] = val
        os.environ["VLLME_ENFORCE_EAGER"] = str(bool(val)).lower()
        os.environ["VLLM_CACHE_ROOT"] = "/runpod-volume/.cache/vllm"
    if os.getenv("TORCHDYNAMO_DISABLE") == "0": print("--- TORCH.COMPILE AND GRAPH CAPTURE IS ENABLED. THIS MAKES COLD-START EXTREMELY LONG, BUT IMPROVES INFERENCE SPEED DRASTICALLY. WE RECOMMEND USING THIS ONLY WITH NETWORK VOLUME, SO IT'S COMPUTED ONLY ONCE AND SHARED BETWEEN WORKERS.")

# ---------------------------------------------------------------------------- #
#                               Functions                                      #
# ---------------------------------------------------------------------------- #
def process_input(input):
    
    print(f"this it the input {input}")
    
    try:
        # Ensure the request is OpenAI compatible and supported
        if "openai_route" not in input or "openai_input" not in input: raise Exception(f"This vLLM image currently supports only OpenAI compatible requests that go to 'your-runpod-endpoint/openai/v1/...' paths")
        # Check if the incoming request is only for pre-warm
        if "prewarm" in input["openai_input"]: return "prewarm"

        print(f"not prewarm")
        
        task_map = { 
            "generate": { "endpoint":["/v1/chat/completions","/v1/completions"] },
            "embedding": { "endpoint":["/v1/embeddings"] }
        }
        if input["openai_route"] not in task_map[engine_args["task"]]["endpoint"]: raise Exception(f"This vLLM endpoint has engine task set to '{engine_args['task']}', so it supports calling only '{task_map[engine_args['task']]['endpoint']}'")
        # Gather and merge Sampling Params
        vllm_supported_sampling_params = ["n","best_of","presence_penalty","frequency_penalty","repetition_penalty","temperature","top_p","top_k","min_p","seed","stop","stop_token_ids","bad_words","include_stop_str_in_output","ignore_eos","max_tokens","min_tokens","logprobs","prompt_logprobs","detokenize","skip_special_tokens","spaces_between_special_tokens","logits_processors","truncate_prompt_tokens","guided_decoding","logit_bias","allowed_token_ids"]
        openai_unsupported_sampling_params = ["bad_words", "detokenize", "allowed_token_ids"]
        input.setdefault("sampling_params", {}) # Ensure the sampling_params dictionary exists
        for p in vllm_supported_sampling_params:
            if input["openai_input"].get(p) is not None: input["sampling_params"][p] = input["openai_input"][p]
        input["openai_input"].update({k: v for k, v in get_args(input.get("sampling_params", {}), "VLLMSP_").items() if k not in openai_unsupported_sampling_params})
        # Check requested model name against supported ones
        supported_model_names = [ os.getenv("OPENAI_SERVED_MODEL_NAME_OVERRIDE", None) or engine_args["served_model_name"], engine_args["model"] ]
        if(input["openai_input"]["model"] not in supported_model_names): raise Exception("Requested model name does not match the model name(s) set in this endpoint.")
        input["openai_input"]["model"] = engine_args["model"] # Change the used model to the one set in engine args, since that is the only supported one from vllm's pov
        # Check what action needs to be taken
        input.setdefault("api", {"action":None,"request":None})
        match input["openai_route"]:
            case "/v1/chat/completions":
                tools = bool(int( os.getenv("VLLMC_ENABLE_AUTO_TOOL_CHOICE","0" )))
                input["api"]["action"] = OpenAIServingChat(engine_client=engine, model_config=api_engine_config, models=api_served_models, response_role=os.getenv("OPENAI_RESPONSE_ROLE", "Assistant"), request_logger=None, chat_template=tokenizer.chat_template, chat_template_content_format="auto", enable_auto_tools=tools, tool_parser=os.getenv("VLLMC_TOOL_CALL_PARSER",None), enable_prompt_tokens_details=False).create_chat_completion
                input["api"]["request"] = ChatCompletionRequest(**input["openai_input"], chat_template=tokenizer.chat_template )
            case "/v1/completions": 
                input["api"]["action"] = OpenAIServingCompletion(engine_client=engine, model_config=api_engine_config, models=api_served_models, request_logger=None).create_completion
                input["api"]["request"] = CompletionRequest(**input["openai_input"])
            case "/v1/embeddings": 
                input["api"]["action"] = OpenAIServingEmbedding(engine_client=engine, model_config=api_engine_config, models=api_served_models, request_logger=None, chat_template=tokenizer.chat_template, chat_template_content_format="auto").create_embedding
                input["api"]["request"] = EmbeddingChatRequest(**input["openai_input"], chat_template=tokenizer.chat_template)
        
        print(f"this it the final input {input}")
        
        return input
    except Exception as err:
        return {"err": str(err), "traceback": traceback.format_exc()}

async def generate(input):
    try: # Classic response
        results = await input["api"]["action"](input["api"]["request"], raw_request=SimpleNamespace(headers={}, state=SimpleNamespace(request_metadata=None), is_disconnected=lambda: False))
        if not input["openai_input"].get("stream", False) or isinstance(results, ErrorResponse):
            yield results.model_dump()
        else: # Streaming response
            batch = []
            batch_size = BatchSize( int(os.getenv("OPENAI_STREAM_BATCH_SIZE",50)), int(os.getenv("OPENAI_STREAM_MIN_BATCH_SIZE",1)), int(os.getenv("OPENAI_STREAM_BATCH_SIZE_GROWTH_FACTOR",3)) )
            raw_openai_output = bool(int(os.getenv("RAW_OPENAI_OUTPUT", 0)))

            async for chunk_str in results:
                if "data" not in chunk_str or "[DONE]" in chunk_str: continue
                data = chunk_str if raw_openai_output else json.loads(chunk_str.removeprefix("data: ").rstrip("\n\n"))
                batch.append(data)
                
                if len(batch) >= batch_size.current_batch_size:
                    output = "\n".join([f"data: {json.dumps(item)}" for item in batch]) + "\n\n"
                    yield output
                    batch.clear()
                    batch_size.update()
            if batch: 
                output = "\n".join([f"data: {json.dumps(item)}" for item in batch]) + "\n\n"
                yield output
            yield "data: [DONE]\n\n"
    except Exception as err:
        yield {"err": str(err), "traceback": traceback.format_exc()}

# ---------------------------------------------------------------------------- #
#                                RunPod Handler                                #
# ---------------------------------------------------------------------------- #
async def handler(job):
    # engine init moved into the handler because of static memory profile feature
    await initialize_engine(job["input"])

    print("Starting RunPod handler and processing input...")
    job_input = process_input(job["input"])
    
    # Prewarm Flashboot request
    if job_input == "prewarm":
        print('─' * 20, "--- Prewarm check completed ---", '─' * 20, sep=os.linesep)
        yield {"warm": True}
    # Normal request
    else:
        gen_start_time = time.time()
        async for result in generate(job_input): yield result
        print('─' * 20, "--- VLLM text generation took %s seconds ---" % (time.time() - gen_start_time), '─' * 20, sep=os.linesep)

# ---------------------------------------------------------------------------- #
#                                 Entrypoint                                   #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler, "concurrency_modifier": concurrency_modifier, "return_aggregate_stream": True})
