# monkey patch llm_engine.py

from vllm.engine.llm_engine import LLMEngine
from vllm.logger import init_logger
import time
logger = init_logger(__name__)

def custom_initialize_kv_caches(self):
    """Initialize the KV cache in the worker(s).

    The workers will determine the number of blocks in both the GPU cache
    and the swap CPU cache.
    """
    start = time.time()
    
    if self.cache_config.num_gpu_blocks_override is not None:
        num_gpu_blocks_override = self.cache_config.num_gpu_blocks_override
        logger.info(
            "--- Using static definition of KV cache memory with %d gpu_blocks.", num_gpu_blocks_override)
        num_gpu_blocks = num_gpu_blocks_override
        num_cpu_blocks = 0
    else:
        num_gpu_blocks, num_cpu_blocks = (
        self.model_executor.determine_num_available_blocks())

    self.cache_config.num_gpu_blocks = num_gpu_blocks
    self.cache_config.num_cpu_blocks = num_cpu_blocks

    self.model_executor.initialize_cache(num_gpu_blocks, num_cpu_blocks)
    elapsed = time.time() - start
    logger.info(("init engine (profile, create kv cache, "
                    "warmup model) took %.2f seconds"), elapsed)

# Monkey patch the method
LLMEngine._initialize_kv_caches = custom_initialize_kv_caches