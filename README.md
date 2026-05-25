# LoRAWeaver Evaluation

This directory contains the code for evaluating layer-wise LoRA adapter retrieval and mixture-based generation.

## Setup

Create a Python environment with CUDA-enabled PyTorch, then install the required packages:

```bash
conda create -n LoRAWeaver python=3.10
conda activate LoRAWeaver
pip install -r requirements.txt
pip install -e peft/
```

The code uses Llama 2 base models from Hugging Face:

- `meta-llama/Llama-2-7b-hf`
- `meta-llama/Llama-2-13b-hf`

Make sure you have accepted the Llama 2 license on Hugging Face and are logged in if the model weights are not already cached:

```bash
huggingface-cli login
```

## Running

From this directory, run:

```bash
python main.py --data_path dataset/combined_test.json
```

By default, this runs mixture evaluation with:

- `--config_path config/config2.json`
- `--res_path results.json`
- `--model_size 13b`
- `--lora_num 3`
- `--prototype_num_samples 20`
- `--max_new_tokens 50`

Example with an explicit output file:

```bash
python main.py \
  --data_path dataset/combined_test.json \
  --res_path results/combined_test_results.json
```

To run a smaller subset for a quick smoke test:

```bash
python main.py \
  --data_path dataset/combined_test.json \
  --max_examples 5 \
  --res_path results/smoke_test.json
```

## Input Format

The evaluation file should be JSON or JSONL. Each example is expected to include:

- `inputs`: the input prompt or task instance
- `targets`: the reference output
- `task`: the task name used to identify the corresponding LoRA adapter
- `metric`: the evaluation metric name
- `domain`: the task domain

## Adapters

LoRA adapters are resolved in `model_list.py` using public Hugging Face adapter names of the form:

```text
Styxxxx/llama2_{model_size}_lora-{task_name}
```

These adapters are public third-party resources and are not included in this directory. If an adapter has already been downloaded, the code first tries to load it from the local Hugging Face cache; otherwise, it downloads it from Hugging Face.

## Outputs

The output JSON contains one record per evaluated example, including the generated answer, reference target, retrieved adapters, layer-wise retrieval results, and run settings.


