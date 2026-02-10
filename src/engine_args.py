import os
import json
import logging
import torch
from torch.cuda import device_count, get_device_properties
from vllm import AsyncEngineArgs



SMALL_GPU_VRAM_BYTES = 40 * (1024 ** 3)
VERY_SMALL_GPU_VRAM_BYTES = 24 * (1024 ** 3)  
TINY_GPU_VRAM_BYTES = 16 * (1024 ** 3)        
GPU_48GB_VRAM_BYTES = 48 * (1024 ** 3)        
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
from utils import convert_limit_mm_per_prompt

RENAME_ARGS_MAP = {
    "MODEL_NAME": "model",
    "MODEL_REVISION": "revision",
    "TOKENIZER_NAME": "tokenizer",
    "MAX_CONTEXT_LEN_TO_CAPTURE": "max_seq_len_to_capture"
}

DEFAULT_ARGS = {
    "disable_log_stats": os.getenv('DISABLE_LOG_STATS', 'False').lower() == 'true',
    "disable_log_requests": os.getenv('DISABLE_LOG_REQUESTS', 'False').lower() == 'true',
    "gpu_memory_utilization": float(os.getenv('GPU_MEMORY_UTILIZATION', 0.95)),
    "pipeline_parallel_size": int(os.getenv('PIPELINE_PARALLEL_SIZE', 1)),
    "tensor_parallel_size": int(os.getenv('TENSOR_PARALLEL_SIZE', 1)),
    "served_model_name": os.getenv('SERVED_MODEL_NAME', None),
    "tokenizer": os.getenv('TOKENIZER', None),
    "skip_tokenizer_init": os.getenv('SKIP_TOKENIZER_INIT', 'False').lower() == 'true',
    "tokenizer_mode": os.getenv('TOKENIZER_MODE', 'auto'),
    "trust_remote_code": os.getenv('TRUST_REMOTE_CODE', 'False').lower() == 'true',
    "download_dir": os.getenv('DOWNLOAD_DIR', None),
    "load_format": os.getenv('LOAD_FORMAT', 'auto'),
    "config_format": os.getenv('CONFIG_FORMAT', 'auto'),
    "dtype": os.getenv('DTYPE', 'auto'),
    "kv_cache_dtype": os.getenv('KV_CACHE_DTYPE', 'auto'),
    "quantization_param_path": os.getenv('QUANTIZATION_PARAM_PATH', None),
    "seed": int(os.getenv('SEED', 0)),
    "max_model_len": int(os.getenv('MAX_MODEL_LEN', 32000)) or None,
    "worker_use_ray": os.getenv('WORKER_USE_RAY', 'False').lower() == 'true',
    "distributed_executor_backend": os.getenv('DISTRIBUTED_EXECUTOR_BACKEND', None),
    "max_parallel_loading_workers": int(os.getenv('MAX_PARALLEL_LOADING_WORKERS', 0)) or None,
    "block_size": int(os.getenv('BLOCK_SIZE', 16)),
    "enable_prefix_caching": os.getenv('ENABLE_PREFIX_CACHING', 'False').lower() == 'true',
    "disable_sliding_window": os.getenv('DISABLE_SLIDING_WINDOW', 'False').lower() == 'true',
    "use_v2_block_manager": os.getenv('USE_V2_BLOCK_MANAGER', 'False').lower() == 'true',
    "swap_space": int(os.getenv('SWAP_SPACE', 4)),
    "cpu_offload_gb": int(os.getenv('CPU_OFFLOAD_GB', 0)),
    "max_num_batched_tokens": int(os.getenv('MAX_NUM_BATCHED_TOKENS', 0)) or None,
    "max_num_seqs": int(os.getenv('MAX_NUM_SEQS', 256)),
    "max_logprobs": int(os.getenv('MAX_LOGPROBS', 20)),
    "revision": os.getenv('REVISION', None),
    "code_revision": os.getenv('CODE_REVISION', None),
    "rope_scaling": os.getenv('ROPE_SCALING', None),
    "rope_theta": float(os.getenv('ROPE_THETA', 0)) or None,
    "tokenizer_revision": os.getenv('TOKENIZER_REVISION', None),
    "quantization": os.getenv('QUANTIZATION', None),
    "enforce_eager": os.getenv('ENFORCE_EAGER', 'False').lower() == 'true',
    "max_context_len_to_capture": int(os.getenv('MAX_CONTEXT_LEN_TO_CAPTURE', 0)) or None,
    "max_seq_len_to_capture": int(os.getenv('MAX_SEQ_LEN_TO_CAPTURE', 8192)),
    "disable_custom_all_reduce": os.getenv('DISABLE_CUSTOM_ALL_REDUCE', 'False').lower() == 'true',
    "tokenizer_pool_size": int(os.getenv('TOKENIZER_POOL_SIZE', 0)),
    "tokenizer_pool_type": os.getenv('TOKENIZER_POOL_TYPE', 'ray'),
    "tokenizer_pool_extra_config": os.getenv('TOKENIZER_POOL_EXTRA_CONFIG', None),
    "enable_lora": os.getenv('ENABLE_LORA', 'False').lower() == 'true',
    "max_loras": int(os.getenv('MAX_LORAS', 1)),
    "max_lora_rank": int(os.getenv('MAX_LORA_RANK', 16)),
    "enable_prompt_adapter": os.getenv('ENABLE_PROMPT_ADAPTER', 'False').lower() == 'true',
    "max_prompt_adapters": int(os.getenv('MAX_PROMPT_ADAPTERS', 1)),
    "max_prompt_adapter_token": int(os.getenv('MAX_PROMPT_ADAPTER_TOKEN', 0)),
    "fully_sharded_loras": os.getenv('FULLY_SHARDED_LORAS', 'False').lower() == 'true',
    "lora_extra_vocab_size": int(os.getenv('LORA_EXTRA_VOCAB_SIZE', 256)),
    "long_lora_scaling_factors": tuple(map(float, os.getenv('LONG_LORA_SCALING_FACTORS', '').split(','))) if os.getenv('LONG_LORA_SCALING_FACTORS') else None,
    "lora_dtype": os.getenv('LORA_DTYPE', 'auto'),
    "max_cpu_loras": int(os.getenv('MAX_CPU_LORAS', 0)) or None,
    "device": os.getenv('DEVICE', 'auto'),
    "ray_workers_use_nsight": os.getenv('RAY_WORKERS_USE_NSIGHT', 'False').lower() == 'true',
    "num_gpu_blocks_override": int(os.getenv('NUM_GPU_BLOCKS_OVERRIDE', 0)) or None,
    "num_lookahead_slots": int(os.getenv('NUM_LOOKAHEAD_SLOTS', 0)),
    "model_loader_extra_config": os.getenv('MODEL_LOADER_EXTRA_CONFIG', None),
    "ignore_patterns": os.getenv('IGNORE_PATTERNS', None),
    "preemption_mode": os.getenv('PREEMPTION_MODE', None),
    "scheduler_delay_factor": float(os.getenv('SCHEDULER_DELAY_FACTOR', 0.0)),
    "enable_chunked_prefill": os.getenv('ENABLE_CHUNKED_PREFILL', None),
    "guided_decoding_backend": os.getenv('GUIDED_DECODING_BACKEND', 'outlines'),
    "speculative_model": os.getenv('SPECULATIVE_MODEL', None),
    "speculative_draft_tensor_parallel_size": int(os.getenv('SPECULATIVE_DRAFT_TENSOR_PARALLEL_SIZE', 0)) or None,
    "enable_expert_parallel": bool(os.getenv('ENABLE_EXPERT_PARALLEL', 'False').lower() == 'true'),
    "num_speculative_tokens": int(os.getenv('NUM_SPECULATIVE_TOKENS', 0)) or None,
    "speculative_max_model_len": int(os.getenv('SPECULATIVE_MAX_MODEL_LEN', 0)) or None,
    "speculative_disable_by_batch_size": int(os.getenv('SPECULATIVE_DISABLE_BY_BATCH_SIZE', 0)) or None,
    "ngram_prompt_lookup_max": int(os.getenv('NGRAM_PROMPT_LOOKUP_MAX', 0)) or None,
    "ngram_prompt_lookup_min": int(os.getenv('NGRAM_PROMPT_LOOKUP_MIN', 0)) or None,
    "spec_decoding_acceptance_method": os.getenv('SPEC_DECODING_ACCEPTANCE_METHOD', 'rejection_sampler'),
    "typical_acceptance_sampler_posterior_threshold": float(os.getenv('TYPICAL_ACCEPTANCE_SAMPLER_POSTERIOR_THRESHOLD', 0)) or None,
    "typical_acceptance_sampler_posterior_alpha": float(os.getenv('TYPICAL_ACCEPTANCE_SAMPLER_POSTERIOR_ALPHA', 0)) or None,
    "qlora_adapter_name_or_path": os.getenv('QLORA_ADAPTER_NAME_OR_PATH', None),
    "disable_logprobs_during_spec_decoding": os.getenv('DISABLE_LOGPROBS_DURING_SPEC_DECODING', None),
    "otlp_traces_endpoint": os.getenv('OTLP_TRACES_ENDPOINT', None),
    "use_v2_block_manager": os.getenv('USE_V2_BLOCK_MANAGER', 'true'),
    "compilation_config": os.getenv('COMPILATION_CONFIG', None),
    "max_cudagraph_capture_size": int(os.getenv('MAX_CUDAGRAPH_CAPTURE_SIZE', 0)) or None,
    "max_cudagraph_capture_sizes": int(os.getenv('MAX_CUDAGRAPH_CAPTURE_SIZE', 0)) or None,
}

limit_mm_env = os.getenv('LIMIT_MM_PER_PROMPT')
if limit_mm_env is not None:
    DEFAULT_ARGS["limit_mm_per_prompt"] = convert_limit_mm_per_prompt(limit_mm_env)

def match_vllm_args(args):
    renamed_args = {RENAME_ARGS_MAP.get(k, k): v for k, v in args.items()}
    matched_args = {k: v for k, v in renamed_args.items() if k in AsyncEngineArgs.__dataclass_fields__}
    return {k: v for k, v in matched_args.items() if v not in [None, "", "None"]}

def get_local_args():
    if not os.path.exists("/local_model_args.json"):
        return {}

    with open("/local_model_args.json", "r") as f:
        local_args = json.load(f)

    if local_args.get("MODEL_NAME") is None:
        logging.warning("Model name not found in /local_model_args.json. There maybe was a problem when baking the model in.")

    logging.info(f"Using baked in model with args: {local_args}")
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    return local_args

def get_engine_args():
    args = DEFAULT_ARGS
    args.update(os.environ)
    args.update(get_local_args())
    args = match_vllm_args(args)

    if args.get("load_format") == "bitsandbytes":
        args["quantization"] = args["load_format"]
    
    num_gpus = device_count()
    if num_gpus > 0:
        if args.get("device") == "auto" or not args.get("device"):
            args["device"] = "cuda"
            logging.info("CUDA available: Setting device to 'cuda'.")
    if num_gpus > 1:
        args["tensor_parallel_size"] = num_gpus
        args["max_parallel_loading_workers"] = None
        if os.getenv("MAX_PARALLEL_LOADING_WORKERS"):
            logging.warning("Overriding MAX_PARALLEL_LOADING_WORKERS with None because more than 1 GPU is available.")
    
    if args.get("kv_cache_dtype") == "fp8_e5m2":
        args["kv_cache_dtype"] = "fp8"
        logging.warning("Using fp8_e5m2 is deprecated. Please use fp8 instead.")
    if os.getenv("MAX_CONTEXT_LEN_TO_CAPTURE"):
        args["max_seq_len_to_capture"] = int(os.getenv("MAX_CONTEXT_LEN_TO_CAPTURE"))
        logging.warning("Using MAX_CONTEXT_LEN_TO_CAPTURE is deprecated. Please use MAX_SEQ_LEN_TO_CAPTURE instead.")
    
    model_name = args.get("model", "").lower()
    if "gpt-oss" in model_name:
        logging.info("Detected GPT-OSS model, applying optimizations......")

       
        small_gpu = False
        very_small_gpu = False  # 16-24 GB (e.g. RTX 4090, 3090)
        tiny_gpu = False       # ≤16 GB
        gpu_48gb_or_more = False
        try:
            if device_count() > 0:
                total_vram = get_device_properties(0).total_memory
                vram_gb = total_vram / (1024 ** 3)
                if total_vram >= GPU_48GB_VRAM_BYTES:
                    gpu_48gb_or_more = True
                if total_vram <= TINY_GPU_VRAM_BYTES:
                    tiny_gpu = True
                    very_small_gpu = True
                    small_gpu = True
                    logging.info(
                        "Tiny GPU detected (%.1f GB VRAM). Using minimal defaults (max_model_len=1024, max_num_seqs=32) for 16GB class.",
                        vram_gb,
                    )
                elif total_vram <= VERY_SMALL_GPU_VRAM_BYTES:
                    very_small_gpu = True
                    small_gpu = True
                    logging.info(
                        "Very small GPU detected (%.1f GB VRAM). Using conservative defaults (max_model_len=2048, max_num_seqs=64) for 16-24GB.",
                        vram_gb,
                    )
                elif total_vram <= SMALL_GPU_VRAM_BYTES:
                    small_gpu = True
                    logging.info(
                        "Small GPU detected (%.1f GB VRAM). Using conservative defaults to avoid OOM.",
                        vram_gb,
                    )
        except Exception as e:
            logging.debug("Could not get GPU memory for small-GPU detection: %s", e)

        if os.getenv('ASYNC_SCHEDULING', 'true').lower() == 'true':
            args["async_scheduling"] = True
            logging.info("Async scheduling enabled for GPT-OSS model.")
        
        if os.getenv('NO_ENABLE_PREFIX_CACHING', 'true').lower() == 'true':
            args["enable_prefix_caching"] = False
            logging.info("Prefix caching disabled for GPT-OSS model.")
        
        if not args.get("max_model_len"):
            if gpu_48gb_or_more:
                default_max_len = 32000
            else:
                default_max_len = 32000
            args["max_model_len"] = default_max_len
            logging.info(
                "Setting default max_model_len to %s for GPT-OSS model.",
                default_max_len,
            )
        if gpu_48gb_or_more and args.get("max_model_len"):
            max_len_val = args["max_model_len"]
            if isinstance(max_len_val, str):
                max_len_val = int(max_len_val) if max_len_val else 8192
            max_len_val = max_len_val or 8192
            if max_len_val < 32000:
                args["max_model_len"] = 32000
                logging.info(
                    "48GB+ GPU: Correcting max_model_len from %s to 32000 for GPT-OSS 20B.",
                    max_len_val,
                )
        if tiny_gpu and args.get("max_num_seqs", 256) > 32:
            args["max_num_seqs"] = 32
            logging.info("Capping max_num_seqs to 32 on tiny GPU (≤16GB VRAM).")
        elif very_small_gpu and args.get("max_num_seqs", 256) > 64:
            args["max_num_seqs"] = 64
            logging.info("Capping max_num_seqs to 64 on very small GPU (16-24GB VRAM).")
        elif gpu_48gb_or_more and not args.get("max_num_seqs"):
            args["max_num_seqs"] = 256
            logging.info("Setting max_num_seqs to 256 on 48GB+ GPU for optimal throughput.")
        elif not small_gpu and not very_small_gpu and not tiny_gpu and not gpu_48gb_or_more and not args.get("max_num_seqs"):
            args["max_num_seqs"] = 128
            logging.info("Setting max_num_seqs to 128 on 40GB GPU for optimal throughput.")
        
        if small_gpu and args.get("gpu_memory_utilization", 0.95) >= 0.94:
            if tiny_gpu:
                args["gpu_memory_utilization"] = 0.80
            elif very_small_gpu:
                args["gpu_memory_utilization"] = 0.85
            else:
                args["gpu_memory_utilization"] = 0.88
            logging.info(
                "Reducing gpu_memory_utilization to %s on small GPU to leave headroom.",
                args["gpu_memory_utilization"],
            )
        elif gpu_48gb_or_more and not os.getenv('GPU_MEMORY_UTILIZATION'):
            args["gpu_memory_utilization"] = 0.92
            logging.info("Setting gpu_memory_utilization to 0.92 on 48GB+ GPU for optimal performance.")
        elif not small_gpu and not very_small_gpu and not tiny_gpu and not gpu_48gb_or_more and not os.getenv('GPU_MEMORY_UTILIZATION'):
            args["gpu_memory_utilization"] = 0.90
            logging.info("Setting gpu_memory_utilization to 0.90 on 40GB GPU for optimal performance.")
        
        max_model_len_val = args.get("max_model_len")
        if isinstance(max_model_len_val, str):
            max_model_len_val = int(max_model_len_val) if max_model_len_val else 8192
        max_model_len_val = max_model_len_val or 8192
        
        env_batched_tokens = os.getenv('MAX_NUM_BATCHED_TOKENS', '')
        if env_batched_tokens and int(env_batched_tokens) > 0:
            target_batched_tokens = max(int(env_batched_tokens), max_model_len_val)
        else:
            if gpu_48gb_or_more:
                target_batched_tokens = max(max_model_len_val * 2, 32000)
            elif not small_gpu and not very_small_gpu and not tiny_gpu:
                target_batched_tokens = max(int(max_model_len_val * 1.5), 24000)
            else:
                target_batched_tokens = max_model_len_val
        
        args["max_num_batched_tokens"] = target_batched_tokens
        logging.info(f"Setting max_num_batched_tokens to {args['max_num_batched_tokens']} for GPT-OSS model (max_model_len={max_model_len_val}).")
        
        if not args.get("max_cudagraph_capture_size"):
            default_cudagraph = (
                256 if tiny_gpu else
                512 if very_small_gpu else
                1024 if small_gpu else
                2048
            )
            cudagraph_size = int(os.getenv('MAX_CUDAGRAPH_CAPTURE_SIZE', default_cudagraph))
            if cudagraph_size > 0:
                args["max_cudagraph_capture_size"] = cudagraph_size
                logging.info(
                    "Setting max_cudagraph_capture_size to %s for GPT-OSS model.",
                    cudagraph_size,
                )
        if (tiny_gpu or very_small_gpu) and args.get("max_seq_len_to_capture"):
            max_len = args.get("max_model_len") or (1024 if tiny_gpu else 2048)
            if args["max_seq_len_to_capture"] > max_len:
                args["max_seq_len_to_capture"] = max_len
                logging.info(
                    "Capping max_seq_len_to_capture to %s on small VRAM GPU.",
                    max_len,
                )
        
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '-i', '0', '--query-gpu=compute_cap', '--format=csv,noheader'], 
                                   capture_output=True, text=True)
            compute_cap = result.stdout.strip()
            compute_major = int(compute_cap.split('.')[0]) if compute_cap else 0
            compute_minor = int(compute_cap.split('.')[1]) if '.' in compute_cap and compute_cap.split('.')[1] else 0
            
            if compute_cap == "10.0":
                if os.getenv('VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8') is None:
                    os.environ["VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8"] = "1"
                    logging.info("Blackwell GPU detected, enabling FlashInfer MXFP4+MXFP8 MoE.")
            elif compute_major < 8 or (compute_major == 8 and compute_minor < 0):
                quant = args.get("quantization", "").lower() if args.get("quantization") else ""
                is_fp4_model = "mxfp4" in quant or "fp4" in quant or "gpt-oss" in model_name
                user_override = os.getenv('ENFORCE_EAGER', '').lower()
                if is_fp4_model and user_override != 'false' and not args.get("enforce_eager"):
                    args["enforce_eager"] = True
                    logging.info(
                        "GPU compute capability %s < 8.0 with FP4/GPT-OSS model. Enabling enforce_eager.",
                        compute_cap,
                    )
            elif compute_major == 8 and compute_minor >= 0:
                user_override = os.getenv('ENFORCE_EAGER', '').lower()
                if user_override == 'true':
                    args["enforce_eager"] = True
                    logging.info("User enabled enforce_eager on Ampere/Ada GPU (compute capability %s).", compute_cap)
                elif user_override != 'false':
                    args["enforce_eager"] = False
                    logging.info("Allowing CUDA graphs on Ampere/Ada GPU (compute capability %s) for optimal performance.", compute_cap)
        except Exception as e:
            logging.debug(f"Could not detect GPU compute capability: {e}")

        if args.get("async_scheduling") is True:
            backend = args.get("distributed_executor_backend")
            if backend == "ray" or backend is None:
                args["distributed_executor_backend"] = "mp"
                logging.info("Async scheduling requires mp/uni/external_launcher; using mp backend.")
    
    backend = args.get("distributed_executor_backend")
    if backend == "mp" and device_count() > 0:
        if args.get("device") == "auto" or not args.get("device"):
            args["device"] = "cuda"
            logging.info("Setting device to 'cuda' for multiprocessing executor backend.")
    
    if "max_num_batched_tokens" in args:
        batched_tokens = args["max_num_batched_tokens"]
        if isinstance(batched_tokens, str):
            batched_tokens = int(batched_tokens) if batched_tokens else None
        if batched_tokens is not None and batched_tokens < 1:
            del args["max_num_batched_tokens"]
            logging.warning("Removed invalid max_num_batched_tokens value (< 1)")
    
    if "device" in args:
        del args["device"]
        
    return AsyncEngineArgs(**args)
