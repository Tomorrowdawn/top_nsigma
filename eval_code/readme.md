This is the evaluation code for top-nsigma sampler.

## Usage

### TLDR

Make sure you are at the root folder. Creating a new environment (py3.10) is recommended.

Make sure you have model weights under `$HF_MODELS_CACHE/{MODEL_NAME}`

```bash
pip install -r requirements.txt
cd eval_code
```

Reasoning:

```bash
python uni_evaluate.py --config_path uni_config/8b.yaml > 8b.log
```

Creative Writing:

First modify the `API_KEY` in `alpaca_evaluator/evaluate_secrete.sh` into your own key. Choose the correct size (8B or 70B) and run the following commands.

```bash
python alpaca.py --config_path uni_config/alpaca_8b.yaml
bash alpaca_evaluator/evaluate_secrete.sh
```

`vllm` version is important (0.5.5), since they had some breaking changes in new updates.

> [!WARNING]
> `datasets` version is tricky. Due to some restrictions, we can only access HuggingFace through mirrors, resulting in an outdated datasets version (which may conflict with vllm). If you want to use a newer datasets version, I personally suggest manually modifying the config file or setting trust_remote_code in script. Note that this solution has never been tested. For full reproduction, manually download the datasets and configure their locations correctly to avoid accessing HuggingFace.

### Script Parameter

```bash
python uni_evaluate.py 
    --config_path # PATH TO YOUR CONFIG
    --gpu_util #vllm's gpu utilization. eg, 0.9
    --cuda # cuda device string. 0,2,4. Tensor parallelization will be automatically triggered if more than one device is provided.
```

```bash
python alpaca.py 
    --config_path # PATH TO YOUR CONFIG
    --gpu_util #vllm's gpu utilization. eg, 0.9
    --cuda # cuda device string. 0,2,4. Tensor parallelization will be automatically triggered if more than one device is provided.
```

## Framework

The code implements an automated and standardized testing framework. It reads the config file, locates corresponding tasks, and automatically runs evaluations with all possible parameter combinations across datasets, models, algorithms and metrics.

### Configuration Files

The framework uses two types of configuration files:

#### 1. Task Configuration (task_config.yaml)

This file defines settings for a specific evaluation task:

- `name`: Name of the task
- `dataset`: Dataset configuration
  - `path`: HuggingFace dataset path
  - `name`: Dataset configuration name 
  - `split`: Dataset split (e.g., "train", "test")
- `preprocessor`: Path to preprocessing function (e.g., "utils.preprocess")
- `output_instruction`: Instructions for formatting model output
- `output_regex`: Regex for extracting answer from output
- `output_norm`: Path to normalization function (optional)

The preprocessing and normalization functions should be in a XX.py file in the same directory.

#### 2. Evaluation Configuration (eval_config.yaml) 

This file defines overall evaluation settings:

- `seed`: Random seed for reproducibility
- `limit`: Number of samples to evaluate (int or float 0-1)
- `save_dir`: Directory for saving results
- `models`: List of models to evaluate
  - `name`: Model name/path (By default and recommended, it will be loaded from HF_MODELS_CACHE)
  - `template`: Prompt template with {question} placeholder
- `algorithms`: List of sampling algorithms
  - `name`: Algorithm name (e.g., "top_p")
  - `params`: VLLM's sampling parameters.
  - `facade_params`: Additional parameters for facade samplers(None if you don't need it).
- `tasks`: List of task config file paths
- `metrics`: List of evaluation metrics (e.g., ExactMatch, PassAtK, Majority)

So, not limited to the exisiting tasks, you can also add your own tasks by creating these two types of files.

##### Alpaca Config

Alpaca config is quite manual because automatical execution might be too costly.  The main difference is you should pass single value for each `params` or `facade_params`. Other options are similar.

### Output results

The results will be saved in the `save_dir` directory like:

```
save_dir/
    leaderboard/
        {DATASET_NAME}_leaderboard.csv
    answers/
        {DATASET_NAME}/
            {MODEL_NAME}/
                {ALGORITHM_NAME}_{PARAMS}.json
```

This offers a quite convenient way to access the results.

### Console Output

The console output consists of logs. We recommend redirecting it to a log file using `>`, otherwise it can be quite noisy.

```bash
python uni_evaluate.py --config_path uni_config/8b.yaml > log.txt
```
