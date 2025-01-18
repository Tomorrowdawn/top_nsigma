import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True, type=str)
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--cpu_offload_gb", type=float, default=0.0)
    parser.add_argument("--gpu_util", type=float, default=0.9)
    args = parser.parse_args()
import os
import time
import numpy ## for MKL compatibility or something. Mystery.

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
from typing import List, Optional, TypedDict
from cattrs import structure, unstructure
from cattrs.v import transform_error
import datasets
from pprint import pprint
import itertools
import importlib.util


# Add environment variable for model directory
MODEL_DIR = os.environ.get("HF_MODELS_CACHE", "")


@define
class DatasetConfig:
    path: str
    name: str
    split: str = "test"


@define
class MetricConfig:
    name: str
    k: int = 1


class KeysConfig(TypedDict):
    question: str
    target: str


@define
class TaskConfig:
    name: str
    dataset: DatasetConfig
    preprocessor: Optional[str] = None
    output_instruction: str = ""
    output_regex: str = ""
    output_norm: Optional[str] = None  # New field for normalization function


@define
class ModelConfig:
    name: str
    template: str
    ###template use a placeholder "question" which would be replaced by the actual question.
    ##basically it defines model-specific prompt format(like system prompt, special tokens, etc.)
    ##it should be explicitly defined to avoid accidental use.


@define
class AlgorithmConfig:
    name: str
    params: dict  ##key1: [v1, v2]/v, key2: [v1, v2]/v...
    ##automatically generate all combinations of params
    ## reuse this class. Subtle abuse.
    facade_params: Optional[dict] = None


@define
class EvalConfig:
    seed: int
    save_dir: str
    models: List[ModelConfig]
    algorithms: List[AlgorithmConfig]
    tasks: List[str]
    metrics: List[MetricConfig]  # New field for metrics
    limit: Optional[int] = None
    fraction: Optional[float] = None


class Task:
    """A class representing a task with questions and targets."""

    def __init__(self, config: TaskConfig, config_dir: str):
        """
        Initialize the Task with configuration.

        Parameters
        ----------
        config : TaskConfig
            Configuration for the task.
        config_dir : str
            Directory of the configuration file.
        """
        self.config = config
        self.config_dir = config_dir
        self.dataset = load_dataset(**unstructure(config.dataset))
        self.preprocessor = self._load_module_function(config.preprocessor)
        self.normalizer = self._load_module_function(config.output_norm)
        if self.preprocessor is not None:
            self.dataset = self.preprocessor(self.dataset)
        self.qt_cache = None

    def _load_module_function(self, function_path: str | None) -> Optional[Callable]:
        """
        Load a function from a module path.

        Parameters
        ----------
        function_path : str | None
            Path to the function in format "module.function".

        Returns
        -------
        Optional[Callable]
            Loaded function or None if path is None.
        """
        if function_path is None:
            return None

        module_name, function_name = function_path.rsplit(".", 1)
        module_path = os.path.join(self.config_dir, f"{module_name}.py")
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None:
            raise ValueError(
                f"Failed to load module {module_name} from {module_path}. Your task config might be wrong."
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore
        return getattr(module, function_name)

    def get_preprocessor(self, preprocessor_path: str | None):
        """Deprecated: Use _load_module_function instead"""
        return self._load_module_function(preprocessor_path)

    def normalize_answer(self, answer: str) -> str:
        """
        Normalize an answer using the configured normalizer.

        Parameters
        ----------
        answer : str
            Raw answer string.

        Returns
        -------
        str
            Normalized answer string.
        """
        if self.normalizer is None:
            return answer
        return self.normalizer(answer)

    def get_question_target_pairs(self, limit: int | float | None = None) -> List[Dict[str, str]]:
        """
        Get all question-target pairs from the dataset.

        Parameters
        ----------
        limit : int | float | None
            Limit the number of samples.

        Returns
        -------
        List[Dict[str, str]]
            A list of dictionaries, each containing a 'question' and 'target'.
        """
        if self.qt_cache is None:
            if limit is None:
                limit = 1.0
            if isinstance(limit, float):
                limit = int(len(self.dataset) * limit)
            limit = min(limit, len(self.dataset))
            samples = random.sample(range(len(self.dataset)), limit)
            selected = self.dataset.select(samples)
            self.qt_cache = [
                {
                    "question": item["question"],  # type: ignore
                    "target": item["target"],  # type: ignore
                }
                for item in selected
            ]
            del self.dataset  # avoid tooooooo large dataset
        return self.qt_cache

    def create_metrics(self, metric_configs: List[MetricConfig]) -> List["Metric"]:
        """
        Create metric objects for this task.

        Parameters
        ----------
        metric_configs : List[MetricConfig]
            List of metric configurations.

        Returns
        -------
        List[Metric]
            List of initialized metric objects.
        """
        metrics = []
        for metric_config in metric_configs:
            metric_name = metric_config.name
            if metric_name == "ExactMatch":
                metrics.append(ExactMatch(self.config.output_regex, self.normalizer))
            elif metric_name == "PassAtK":
                metrics.append(PassAtK(metric_config.k, self.config.output_regex, self.normalizer))
            elif metric_name == "Majority":
                metrics.append(Majority(self.config.output_regex, None, self.normalizer))
            else:
                raise ValueError(f"Unsupported metric: {metric_name}")
        return metrics


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
        model_path = os.path.join(MODEL_DIR, model_name)
        self.model = vllm.LLM(
            model_path,
            enforce_eager=True,
            enable_prefix_caching=True,
            tensor_parallel_size=NUM_GPUS,
            cpu_offload_gb=CPU_OFFLOAD_GB,
            gpu_memory_utilization=GPU_UTIL,
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
        self.sampling_params.stop = ["<|eot_id|>", "<|endoftext|>", "<|im_end|>"]
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


class Metric(ABC):
    """Abstract base class for metrics."""

    def __init__(self, output_regex: str, normalizer: Optional[Callable[[str], str]] = None):
        """
        Initialize the metric.

        Parameters
        ----------
        output_regex : str
            Regular expression for extracting answers.
        normalizer : Optional[Callable[[str], str]]
            Function to normalize answers before comparison.
        """
        self.output_regex = output_regex
        self.normalizer = normalizer

    def extract_answer(self, text: str) -> str:
        """
        Extract and normalize an answer from text.

        Parameters
        ----------
        text : str
            Raw text containing the answer.

        Returns
        -------
        str
            Extracted and normalized answer.
        """
        match = re.search(self.output_regex, text, re.DOTALL)
        extracted = match.group(1) if match else text
        extracted = extracted.strip(" .")  ##remove trailing dot and space
        if self.normalizer:
            return self.normalizer(extracted)
        return extracted

    @abstractmethod
    def compute(self, answers: List[List[str]], targets: List[str]) -> float:
        """
        Compute the metric for given answers and targets.

        Parameters
        ----------
        answers : List[List[str]]
            List of lists of model-generated answers.
        targets : List[str]
            List of target answers.

        Returns
        -------
        float
            Computed metric value.
        """
        pass


class ExactMatch(Metric):
    """Exact Match metric for single answer evaluation."""

    def __init__(self, output_regex: str, normalizer: Optional[Callable[[str], str]] = None):
        self.output_regex = output_regex
        # print("output_regex: ", output_regex)
        self.normalizer = normalizer

    def compute(self, answers: List[List[str]], targets: List[str]) -> float:
        """
        Compute the Exact Match score.

        Parameters
        ----------
        answers : List[List[str]]
            List of lists of model-generated answers (each inner list should contain only one answer).
        targets : List[str]
            List of target answers.

        Returns
        -------
        float
            Exact Match score.
        """
        extracted_answers = [[self.extract_answer(ans) for ans in answer_set] for answer_set in answers]
        print("sample extracted answers: ")
        pprint(extracted_answers[:3])
        if len(extracted_answers[0]) != 1:
            warnings.warn(
                "ExactMatch is designed for single answer evaluation. The model may not be calibrated properly."
            )
        correct = sum(ans[0].strip() == target.strip() for ans, target in zip(extracted_answers, targets))
        return correct / len(targets)


class PassAtK(Metric):
    """Pass@K metric for multiple answer evaluation."""

    def __init__(self, k: int, output_regex: str, normalizer: Optional[Callable[[str], str]] = None):
        self.k = k
        self.output_regex = output_regex
        self.normalizer = normalizer

    def compute(self, answers: List[List[str]], targets: List[str]) -> float:
        """
        Compute the Pass@K score.

        Parameters
        ----------
        answers : List[List[str]]
            List of lists of model-generated answers.
        targets : List[str]
            List of target answers.

        Returns
        -------
        float
            Pass@K score.
        """
        extracted_answers = [[self.extract_answer(ans) for ans in answer_set] for answer_set in answers]

        def pass_k(anses, target):
            n = len(anses)
            c = sum(ans.strip() == target.strip() for ans in anses)
            wrong_samples = math.comb(n - c, self.k)
            total_samples = math.comb(n, self.k)
            return 1 - wrong_samples / total_samples

        pass_k_scores = [pass_k(anses, target) for anses, target in zip(extracted_answers, targets)]
        return sum(pass_k_scores) / len(pass_k_scores)


class Majority(Metric):
    """Majority metric for multiple answer evaluation."""

    def __init__(
        self,
        output_regex: str | None,
        k: int | None = None,
        normalizer: Optional[Callable[[str], str]] = None,
    ):
        self.output_regex = output_regex
        self.k = k
        self.normalizer = normalizer

    def compute(self, answers: List[List[str]], targets: List[str]) -> float:
        """
        Compute the Majority score.

        Parameters
        ----------
        answers : List[List[str]]
            List of lists of model-generated answers.
        targets : List[str]
            List of target answers.

        Returns
        -------
        float
            Majority score.
        """
        extracted_answers = [[self.extract_answer(ans) for ans in answer_set] for answer_set in answers]

        def get_majority_answer(anses):
            return max(set(anses), key=anses.count)

        if self.k is None:
            k = len(extracted_answers[0])
        else:
            k = self.k
        majority_answers = [get_majority_answer(anses[:k]) for anses in extracted_answers]
        correct = sum(maj.strip() == target.strip() for maj, target in zip(majority_answers, targets))
        return correct / len(targets)


def load_config(config_path: str) -> tuple[EvalConfig | TaskConfig, str]:
    """
    Load configuration from a YAML file.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    EvalConfig | TaskConfig
        Loaded configuration object.
    """
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    if "tasks" in config_dict:
        return structure(config_dict, EvalConfig), os.path.dirname(config_path)
    else:
        return structure(config_dict, TaskConfig), os.path.dirname(config_path)


def save_answers(
    eval_config: EvalConfig,
    task_config: TaskConfig,
    questions: List[str],
    targets: List[str],
    answers: List[List[str]],
    model: VLLMModel,
    algorithm: AlgorithmConfig,
) -> str:
    """
    Save model-generated answers to a JSON file.

    Parameters
    ----------
    eval_config : EvalConfig
        Evaluation configuration.
    task_config : TaskConfig
        Task configuration.
    questions : List[str]
        List of questions.
    targets : List[str]
        List of target answers.
    answers : List[List[str]]
        List of lists of model-generated answers.
    model : VLLMModel
        The model used for evaluation.
    algorithm : AlgorithmConfig
        The algorithm configuration used for evaluation.

    Returns
    -------
    str
        Path to the saved answers file.
    """
    results_data = [
        {"question": q, "target": t, "answers": a} for q, t, a in zip(questions, targets, answers)
    ]

    metadata = {
        "task_name": task_config.name,
        "dataset": task_config.dataset.path,
        "split": task_config.dataset.split,
        "model": model.model_name,
        "algorithm": unstructure(algorithm),
        "timestamp": datetime.now().isoformat(),
    }

    save_data = {"metadata": metadata, "results": results_data}

    save_dir = os.path.join(eval_config.save_dir, "answers", task_config.name, model.model_name)
    os.makedirs(save_dir, exist_ok=True)

    algo_name = algorithm.name
    params_string = "_".join(f"{k}_{v}" for k, v in algorithm.params.items())
    if algorithm.facade_params:
        params_string += "_" + "_".join(f"{k}_{v}" for k, v in algorithm.facade_params.items())
    filename = f"{algo_name}_{params_string}.json"

    full_path = os.path.join(save_dir, filename)

    with open(full_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Answers saved to {full_path}")
    return full_path


def update_leaderboard(
    eval_config: EvalConfig,
    task_config: TaskConfig,
    model: VLLMModel,
    algorithm: AlgorithmConfig,
    metrics: List[Metric],
    results: List[float],
):
    """
    Update the leaderboard with the evaluation results.

    Parameters
    ----------
    eval_config : EvalConfig
        Evaluation configuration.
    task_config : TaskConfig
        Task configuration.
    model : VLLMModel
        The model used for evaluation.
    algorithm : AlgorithmConfig
        The algorithm configuration used for evaluation.
    metrics : List[Metric]
        The metrics used for evaluation.
    results : List[float]
        The evaluation results for each metric.
    """
    leaderboard_dir = os.path.join(eval_config.save_dir, "leaderboard")
    os.makedirs(leaderboard_dir, exist_ok=True)
    leaderboard_path = os.path.join(leaderboard_dir, f"{task_config.name}_leaderboard.csv")

    new_entry = {
        "task": task_config.name,
        "model": model.model_name,
        "algorithm": algorithm.name,
        "timestamp": datetime.now().isoformat(),
        "params": json.dumps(algorithm.params),
        "facade_params": json.dumps(algorithm.facade_params) if algorithm.facade_params else None,
    }

    for metric, result in zip(metrics, results):
        new_entry[f"{metric.__class__.__name__}_score"] = result

    if os.path.exists(leaderboard_path):
        leaderboard_df = pd.read_csv(leaderboard_path)
        leaderboard_df = pd.concat([leaderboard_df, pd.DataFrame([new_entry])], ignore_index=True)
    else:
        leaderboard_df = pd.DataFrame([new_entry])

    # Sort by the first metric's score in descending order
    first_metric_name = f"{metrics[0].__class__.__name__}_score"
    leaderboard_df = leaderboard_df.sort_values(first_metric_name, ascending=False).reset_index(drop=True)
    leaderboard_df.to_csv(leaderboard_path, index=False)
    print(f"Leaderboard updated at {leaderboard_path}")


def param_combinations(algorithm):
    """
    Generate parameter combinations for an algorithm using an iterator.

    Parameters
    ----------
    algorithm : AlgorithmConfig
        Algorithm configuration.

    Yields
    ------
    AlgorithmConfig
        Algorithm configuration with a specific parameter combination.
    """
    params_items = list(algorithm.params.items())
    facade_params_items = list(algorithm.facade_params.items()) if algorithm.facade_params else []

    # Create a base combination with only params
    base_combinations = itertools.product(
        *[[(key, value) for value in values] for key, values in params_items]
    )

    # If facade_params is not empty, create combinations with facade_params
    if facade_params_items:
        for base_combo in base_combinations:
            for facade_combo in itertools.product(
                *[[(key, value) for value in values] for key, values in facade_params_items]
            ):
                params = dict(base_combo)
                facade_params = dict(facade_combo)
                yield AlgorithmConfig(name=algorithm.name, params=params, facade_params=facade_params)
    else:
        # If facade_params is empty, yield combinations with only params
        for base_combo in base_combinations:
            params = dict(base_combo)
            yield AlgorithmConfig(name=algorithm.name, params=params, facade_params=None)


def load_tasks(task_paths, metrics) -> list[tuple[TaskConfig, Task]]:
    """
    Load all tasks in advance.

    Parameters
    ----------
    task_paths : List[str]
        List of paths to task configuration files.
    metrics : List[Metric]
        List of metrics to be used for evaluation.

    Returns
    -------
    List[Tuple[TaskConfig, Task]]
        List of tuples containing task configurations and initialized tasks.
    """
    tasks = []
    for task_path in task_paths:
        task_config, config_dir = load_config(task_path)
        assert isinstance(task_config, TaskConfig)
        task = Task(task_config, config_dir)
        tasks.append((task_config, task))
    return tasks


def evaluate(eval_config_path: str):
    """
    Run evaluations based on the evaluation configuration.

    Parameters
    ----------
    eval_config_path : str
        Path to the evaluation configuration YAML file.
    """
    eval_config, eval_config_dir = load_config(eval_config_path)
    print(eval_config)
    assert isinstance(eval_config, EvalConfig)
    seed = eval_config.seed

    # Load all tasks in advance
    tasks = load_tasks(eval_config.tasks, eval_config.metrics)

    for model_config in eval_config.models:
        model = VLLMModel(model_config.name, seed=seed)

        for algorithm in eval_config.algorithms:
            for algorithm_config in param_combinations(algorithm):
                model.set_config(algorithm_config)

                for task_config, task in tasks:
                    random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    np.random.seed(seed)

                    # Create metrics for this task
                    metrics = task.create_metrics(eval_config.metrics)

                    # Print progress notification
                    print("-----------------------------------")
                    print(f"Evaluating task: {task_config.name}")
                    print(f"Model: {model_config.name}")
                    print(f"Algorithm: {algorithm_config.name}")
                    print(f"Parameters: {algorithm_config.params}")
                    print("-----------------------------------")

                    # Apply limit to the number of samples
                    limit = eval_config.limit
                    if limit is None:
                        limit = eval_config.fraction
                    question_target_pairs = task.get_question_target_pairs(limit)
                    questions = [pair["question"] for pair in question_target_pairs]
                    targets = [pair["target"] for pair in question_target_pairs]

                    # Combine original questions with output instructions
                    formatted_questions = [f"{q}\n\n{task_config.output_instruction}" for q in questions]

                    # Use model_config.template to format the prompts
                    prompts = [model_config.template.format(question=q) for q in formatted_questions]
                    answers = model.run(prompts)
                    # print("sample answers: ")
                    # pprint(answers[:3])
                    print("sample targets: ")
                    pprint(targets[:3])

                    # Save raw answers
                    save_answers(
                        eval_config, task_config, questions, targets, answers, model, algorithm_config
                    )

                    # Compute metrics and update leaderboard
                    results = [metric.compute(answers, targets) for metric in metrics]
                    update_leaderboard(eval_config, task_config, model, algorithm_config, metrics, results)

                    print(f"Evaluation results for {task_config.name} (using {limit} samples):")
                    for metric, result in zip(metrics, results):
                        print(f"{metric.__class__.__name__}: {result}")

        del model
        gc.collect()


class ResultAnalyzer:
    """A class to analyze and display evaluation results."""

    @staticmethod
    def load_results(results_dir: str) -> pd.DataFrame:
        """
        Load all results from a directory and convert them to a DataFrame.
        Also return the full metadata for each result, keyed by a unique identifier.

        Parameters
        ----------
        results_dir : str
            Path to the directory containing result JSON files.

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, Dict]]
            DataFrame containing all evaluation results and a dictionary of full metadata,
            where keys are unique identifiers and values are metadata dictionaries.
        """
        results = []
        for filename in os.listdir(results_dir):
            if filename.endswith(".json"):
                with open(os.path.join(results_dir, filename)) as f:
                    data = json.load(f)
                    metadata = data["metadata"]
                    unique_id = f"{metadata['task_name']}_{metadata['model']}_{metadata['algorithm']['name']}_{metadata['timestamp']}"
                    results.append(
                        {
                            "id": unique_id,
                            "task": metadata["task_name"],
                            "model": metadata["model"],
                            "algorithm": metadata["algorithm"]["name"],
                            "timestamp": metadata["timestamp"],
                            "results": data["results"],
                            "params": json.dumps(metadata["algorithm"]["params"]),
                            "facade_params": json.dumps(metadata["algorithm"]["facade_params"])
                            if metadata["algorithm"]["facade_params"]
                            else None,
                            "metadata": metadata,
                        }
                    )
        return pd.DataFrame(results)

    @staticmethod
    def display_results(df: pd.DataFrame):
        """
        Display the results in a formatted table.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the evaluation results.
        """
        print(df.to_string(index=False))


if __name__ == "__main__":
    try:
        evaluate(args.config_path)
    except Exception as e:
        print(e)
        print("-----------------------------------")
        print(transform_error(e))
        raise e
        exit(1)
"""
Configuration File Structure:

1. Task Configuration (task_config.yaml):
   This file defines the settings for a specific evaluation task.

   name: str
     Name of the task.

   dataset: Dict
     path: str
       HuggingFace dataset path.
     name: str
       Specific configuration name of the dataset.
     split: str
       Dataset split to use (e.g., "train", "test", "validation").

   preprocessor: str
     Path to a preprocessing function (e.g., "utils.preprocess").
     The function should accept a dataset object and return a modified dataset.
     The utils.py file should be in the same directory as the task config file.

   metric: Dict
     name: str
       Name of the evaluation metric (e.g., ExactMatch, PassAtK, Majority).
     k: Optional[int]
       Parameter for metrics that require a k value (e.g., PassAtK).

   output_instruction: str
     Instructions for formatting the model output.

   output_regex: str
     Regular expression for extracting the answer from model output.
   output_norm: str | None
     Path to a normalization function (e.g., "utils.normalize"). None stands for lambda x: x.
     The function should accept a string and return a normalized string.
     The utils.py file should be in the same directory as the task config file.

2. Evaluation Configuration (eval_config.yaml):
   This file defines the overall settings for running evaluations.

   seed: int
     Random seed for reproducibility.

   limit: Union[int, float]
     Limit on the number of samples to evaluate.
     Can be an integer (exact number) or a float between 0 and 1 (fraction of dataset).

   save_dir: str
     Directory path for saving evaluation results.

   models: List[Dict]
     List of models to evaluate.
     Each model is defined by:
       name: str
         Name or path of the model.
       template: str
         Prompt template for the model, using "question" as a placeholder.

   algorithms: List[Dict]
     List of sampling algorithms to use in evaluation.
     Each algorithm is defined by:
       name: str
         Name of the algorithm (e.g., "top_p", "top_k").
       params: Dict
         Parameters for the algorithm (e.g., temperature, max_tokens).
         For each parameter, a list of values is provided.
         All combinations of the parameters will be used.
       facade_params: Optional[Dict]
         Additional parameters for custom sampling methods.

   tasks: List[str]
     List of paths to task configuration files to be evaluated.

Example:
eval_config.yaml:
  seed: 42
  limit: 1000
  save_dir: "./evaluation_results"
  models:
    - name: "llama-13b"
      template: "<|BOS|>Answer the following question: {question}<|EOS|>"
    - name: "llama-70b"
      template: "Please provide an answer to this question: {question}"
  algorithms:
    - name: "top_p"
      params:
        temperature: [0.7, 0.8, 0.9]
        max_tokens: [100]
        top_p: [0.95, 0.96, 0.97]###will generate 3*1*3=9 combinations
  tasks:
    - "tasks/squad.yaml"
    - "tasks/triviaqa.yaml"
  metrics:
    - name: ExactMatch
    - name: Majority

task_config.yaml (e.g., squad.yaml):
  name: "SQuAD"
  dataset:
    path: "squad"
    name: "plain_text"
    split: "validation"
  preprocessor: "utils.preprocess_squad"
  output_norm: "utils.normalize_answer"
  output_instruction: "Please provide your final answer in the form of Answer: (answer), for example, Answer: (A)."
  output_regex: Answer: (.*)
"""
