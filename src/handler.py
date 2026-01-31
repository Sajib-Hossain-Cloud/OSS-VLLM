import os
import runpod
from utils import JobInput
from engine import vLLMEngine, OpenAIvLLMEngine

vllm_engine = None
openai_engine = None

def get_engines():
    global vllm_engine, openai_engine
    if vllm_engine is None:
        vllm_engine = vLLMEngine()
        openai_engine = OpenAIvLLMEngine(vllm_engine)
    return vllm_engine, openai_engine

async def handler(job):
    engine, openai_eng = get_engines()
    job_input = JobInput(job["input"])
    results_engine = openai_eng if job_input.openai_route else engine
    results_generator = results_engine.generate(job_input)
    async for batch in results_generator:
        yield batch

if __name__ == '__main__':
    vllm_engine, openai_engine = get_engines()
    runpod.serverless.start(
        {
            "handler": handler,
            "concurrency_modifier": lambda x: vllm_engine.max_concurrency,
            "return_aggregate_stream": True,
        }
    )
