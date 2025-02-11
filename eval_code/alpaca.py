import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True, type=str)
    parser.add_argument("--cuda", type=str, default="7")
    parser.add_argument("--cpu_offload_gb", type=float, default=0.0)
    parser.add_argument("--gpu_util", type=float, default=0.9)
    args = parser.parse_args()
import os
import time
import numpy  ## for MKL compatibility or something. Mystery.

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    NUM_GPUS = len(args.cuda.split(","))
    CPU_OFFLOAD_GB = args.cpu_offload_gb
    GPU_UTIL = args.gpu_util
import yaml
from datasets import load_dataset
from typing import List, Dict, Any, Callable
from abc import ABC, abstractmethod
import json
from datetime import datetime
import os
import pandas as pd
from hack import hack_vllm, get_sampler_raw, recover_sampler_raw
from my_sampler import FacadeSampler
import vllm
import gc
import math
import warnings
import textwrap
import random
import torch
import numpy as np
from attrs import define, asdict
import re
from typing import List, Optional, TypedDict, Union
from cattrs import structure, unstructure
from cattrs.v import transform_error
import datasets
from pprint import pprint
import itertools
import importlib.util

MODEL_DIR = os.getenv("HF_MODELS_CACHE")
assert MODEL_DIR is not None, "HF_MODELS_CACHE is not set"


@define
class AlgorithmConfig:
    name: str
    params: dict
    facade_params: Optional[dict] = None


@define
class Config:
    """
    Configuration class for model and algorithms.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility
    model_name : str
        Name of the model to use
    algorithms : List[AlgorithmConfig]
        List of algorithm configurations
    """

    seed: int
    model_name: str
    limit: int
    save_dir: str
    template: str  ## XXX{instruction}YYY
    algorithms: List[AlgorithmConfig]

    """
    Example config template:
    seed: 42
    model_name: "Llama-3-8B-Instruct"
    save_dir: "./results"
    template: "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful expert problem solver. Please strictly follow the user's instructions, especially the output format.<|eot_id|><|start_header_id|>user<|end_header_id|>{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    algorithms:
      - name: "greedy"
        params:
          temperature: 1.0
          max_tokens: 2048
      - name: "top_nsigma"
        params:
          temperature: 1.5
          max_tokens: 2048
        facade_params:
          top_nsigma: 0.8
    """


def load_config(config_path: str) -> Config:
    """
    Load configuration from a YAML file.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    Config
        Parsed configuration object.
    """
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)

    return structure(raw_config, Config)


class AlpacaOutput(TypedDict):
    instruction: str
    output: str
    generator: str
    dataset: str


def load_alpaca_eval(template: str, seed: int, limit: int):
    # dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    # dataset = dataset.shuffle(seed=seed).select(range(limit))
    dataset = []
    with open("./alpaca_evaluator/data.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            # if data["source"] == "writing_prompts":
            dataset.append({"instruction": data["prompt"], "dataset": data["source"]})
    dataset = datasets.Dataset.from_pandas(pd.DataFrame(dataset))
    limit = min(limit, len(dataset))
    dataset = dataset.shuffle(seed=seed).select(range(limit))

    def process_doc(doc):
        doc["input"] = template.format(instruction=doc["instruction"])
        return doc

    dataset = dataset.map(process_doc)
    return dataset  ## fields: [instruction, output, generator, dataset, input]


class Model(ABC):
    """Abstract base class for models."""

    @abstractmethod
    def run(self, prompts: List[str]) -> List[List[str]]:
        """
        Run the model on a list of questions.

        Parameters
        ----------
        questions : List[str]
            List of questions to run the model on.

        Returns
        -------
        List[List[str]]
            List of lists of answers for each question.
        """
        pass


class VLLMModel(Model):
    def __init__(self, model_name: str, seed: int):
        """
        Initialize the VLLMModel.

        Parameters
        ----------
        model_name : str
            Name of the model.
        """
        model_path = os.path.join(MODEL_DIR, model_name)  # type:ignore
        self.model = vllm.LLM(
            model_path,
            enforce_eager=True,
            enable_prefix_caching=True,
            tensor_parallel_size=NUM_GPUS,
            gpu_memory_utilization=GPU_UTIL,
            cpu_offload_gb=CPU_OFFLOAD_GB,
            max_model_len=8 * 1024,
        )
        self.raw_sampler = get_sampler_raw(self.model)
        self.model_name = model_name
        self.seed = seed

    def set_config(self, config: AlgorithmConfig):
        """
        Set the configuration for the model.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary containing sampling parameters.
        """
        if config.facade_params:
            print("Facade params: ", config.facade_params)
            sampler = FacadeSampler(**config.facade_params)
            hack_vllm(self.model, sampler)
        else:
            recover_sampler_raw(self.model, self.raw_sampler)
        self.sampling_params = vllm.SamplingParams(**config.params, seed=self.seed)

    def run(self, prompts: List[str]) -> List[List[str]]:
        """
        Run the model on a list of questions.

        Parameters
        ----------
        questions : List[str]
            List of questions to run the model on.

        Returns
        -------
        List[List[str]]
            List of lists of answers for each question.
        """
        self.sampling_params.stop = ["<|eot_id|>"]
        print("Input samples:")
        pprint(prompts[:3])
        start = time.time()
        outputs = self.model.generate(prompts, self.sampling_params)
        end = time.time()
        print("Time taken: ", end - start)
        output_tokens = [[len(seq_out.token_ids) for seq_out in output.outputs] for output in outputs]
        total_tokens = sum(sum(tokens) for tokens in output_tokens)
        avg_tokens = total_tokens / (len(outputs) * len(outputs[0].outputs))
        print("Avg tokens: ", avg_tokens)
        print("Throughput: ", f"{total_tokens / (end - start):.2f}", " token/s")
        output_texts = [[seq_out.text for seq_out in output.outputs] for output in outputs]
        print("Output samples:")
        samples = output_texts[:3]
        if isinstance(samples[0], list):
            pprint([sample[0] for sample in samples])
        else:
            pprint(samples)
        return output_texts


class AlpacaEvaluator:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        os.makedirs(self.config.save_dir, exist_ok=True)
        self.seed = self.config.seed
        self.model = VLLMModel(self.config.model_name, self.seed)
        self.preamble()
        self.dataset = load_alpaca_eval(self.config.template, self.seed, self.config.limit)

    def run(self):
        for algo in self.config.algorithms:
            results = self.run_algo(algo)
            path = os.path.join(self.config.save_dir, f"{algo.name}.json")
            with open(path, "w") as f:
                json.dump(results, f)

    def preamble(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def run_algo(self, algo: AlgorithmConfig) -> List[AlpacaOutput]:
        self.preamble()
        print("Evaluating...")
        print(algo.name)
        self.model.set_config(algo)
        inputs = list(self.dataset["input"])
        outputs = self.model.run(inputs)
        return [
            AlpacaOutput(
                instruction=item["instruction"],
                output=output[0],
                generator=algo.name,
                dataset=item["dataset"],
            )
            for output, item in zip(outputs, self.dataset)
        ]


if __name__ == "__main__":
    evaluator = AlpacaEvaluator(args.config_path)
    evaluator.run()
