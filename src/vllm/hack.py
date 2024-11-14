import vllm
from vllm.executor.gpu_executor import GPUExecutor
from vllm.worker.model_runner import GPUModelRunnerBase

def hack_vllm(llm:vllm.LLM, sampler):
    ex:GPUExecutor = llm.llm_engine.model_executor
    worker:GPUModelRunnerBase = ex.driver_worker.model_runner
    worker.model.sampler = sampler

def get_sampler(model):
    llm = model.model
    ex:GPUExecutor = llm.llm_engine.model_executor
    worker:GPUModelRunnerBase = ex.driver_worker.model_runner
    sampler = worker.model.sampler
    return sampler

def recover_sampler(model, sampler):
    hack_vllm(model.model, sampler)

def get_sampler_raw(model):
    llm = model
    ex:GPUExecutor = llm.llm_engine.model_executor
    worker:GPUModelRunnerBase = ex.driver_worker.model_runner
    sampler = worker.model.sampler
    return sampler

def recover_sampler_raw(model, sampler):
    hack_vllm(model, sampler)