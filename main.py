import json
import os
import random
from functools import lru_cache

import numpy as np
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import snapshot_download
from peft import PeftModel
from peft.utils.constants import CONFIG_NAME
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_list import (
    build_lora_adapter_ref,
    get_base_model_path,
    normalize_model_size,
)
from utils.instructor_retrieval import (
    build_layerwise_direct_lora_mapping,
    collect_layerwise_lora_paths,
    initialize_index_dw,
    perform_search_dw_surgery,
)
from utils.prompter import Prompter


prompter = Prompter("alpaca")


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


@lru_cache(maxsize=None)
def resolve_adapter_path(adapter_ref):
    def has_adapter_config(path):
        return os.path.isfile(os.path.join(path, CONFIG_NAME))

    if os.path.isdir(adapter_ref):
        if not has_adapter_config(adapter_ref):
            raise FileNotFoundError(
                f"Local adapter directory '{adapter_ref}' does not contain '{CONFIG_NAME}'."
            )
        return adapter_ref

    try:
        return snapshot_download(repo_id=adapter_ref, local_files_only=True)
    except Exception:
        return snapshot_download(repo_id=adapter_ref)


def build_adapter_name(task_name, model_size):
    return build_lora_adapter_ref(model_size, task_name)


def get_base_model_name_or_path(model_size):
    return get_base_model_path(model_size)


def load_base_model(model_size):
    model_path = get_base_model_name_or_path(model_size)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    model.bfloat16()
    model.eval()
    return model, tokenizer


def load_peft_model(lora_paths, base_model):
    if not lora_paths:
        raise ValueError("No LoRA adapters were provided.")

    peft_model = None
    adapter_names = []
    for idx, lora_path in enumerate(lora_paths):
        adapter_name = f"adapter{idx}"
        local_lora_path = resolve_adapter_path(lora_path)
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(
                base_model,
                local_lora_path,
                adapter_name,
                local_files_only=True,
            )
        else:
            peft_model.load_adapter(
                local_lora_path,
                adapter_name,
                local_files_only=True,
            )
        adapter_names.append(adapter_name)

    peft_model.set_adapter(adapter_names)
    peft_model.to("cuda" if torch.cuda.is_available() else "cpu")
    peft_model.eval()
    return peft_model, adapter_names


def init_vector_db(config_path, base_model, tokenizer, model_size, prototype_num_samples):
    with open(config_path, "r", encoding="utf-8") as file:
        lora_configs = json.load(file)

    initialize_index_dw(
        lora_configs,
        model_size=model_size,
        base_model=base_model,
        tokenizer=tokenizer,
        prototype_num_samples=prototype_num_samples,
    )


def select_dataset_split(dataset_obj, requested_split=None):
    if isinstance(dataset_obj, Dataset):
        return dataset_obj
    if requested_split is not None:
        return dataset_obj[requested_split]
    if "train" in dataset_obj:
        return dataset_obj["train"]
    return dataset_obj[list(dataset_obj.keys())[0]]


def load_eval_dataset(data_path, split=None, max_examples=None):
    if data_path is None:
        raise ValueError("data_path is required.")

    if data_path.endswith((".json", ".jsonl")):
        dataset_obj = load_dataset("json", data_files=data_path)
    else:
        dataset_obj = load_dataset(data_path)

    eval_data = select_dataset_split(dataset_obj, requested_split=split)
    eval_data = eval_data.map(
        lambda row: {
            "full_prompt": prompter.generate_prompt(row["inputs"], "", ""),
        }
    )

    if max_examples is not None:
        eval_data = eval_data.select(range(min(int(max_examples), len(eval_data))))

    return eval_data


def decode_response(tokenizer, output_ids):
    text = tokenizer.decode(output_ids, skip_special_tokens=True)
    return text.strip().split("### Response:\n")[-1].strip()


def build_best_selection_layer_hits(layer_hits, task_name, model_size):
    oracle_adapter = build_adapter_name(task_name, model_size)
    return {
        layer_name: [{"lora_path": oracle_adapter, "score": 1.0}]
        for layer_name in layer_hits
    }


def generate_mixture(
    base_model,
    tokenizer,
    prompt_text,
    layer_hits,
    max_new_tokens,
    use_softmax,
    softmax_temperature,
):
    device = next(base_model.parameters()).device
    inputs = tokenizer(
        [prompt_text],
        max_length=512,
        return_tensors="pt",
        padding=True,
    ).to(device)

    lora_paths = collect_layerwise_lora_paths(layer_hits)
    if not lora_paths:
        raise ValueError("mixture did not retrieve any layerwise LoRA adapters.")

    lora_mapping = build_layerwise_direct_lora_mapping(
        layer_hits=layer_hits,
        lora_paths=lora_paths,
        batch_size=inputs["input_ids"].shape[0],
        use_softmax=use_softmax,
        softmax_temperature=softmax_temperature,
    )

    peft_model, adapter_names = load_peft_model(lora_paths, base_model)
    try:
        outputs = peft_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            merging_type="mixture",
            lora_mapping=lora_mapping,
        )
    finally:
        for adapter_name in adapter_names:
            peft_model.delete_adapter(adapter_name)
        del peft_model
        torch.cuda.empty_cache()

    return decode_response(tokenizer, outputs[0]), lora_paths


def eval_datasets(
    data_path=None,
    res_path="results.json",
    config_path="config/config2.json",
    eval_type="mixture",
    lora_num=3,
    batch_size=1,
    ood=False,
    best_selection=False,
    model_size="13b",
    activation_coarse_top_m=8,
    prototype_num_samples=20,
    mixture_softmax=False,
    mixture_temperature=10.0,
    max_examples=None,
    max_new_tokens=50,
    benchmark_split=None,
):
    if eval_type != "mixture":
        raise ValueError("This cleaned main.py only supports eval_type='mixture'.")
    if int(batch_size) != 1:
        raise ValueError("mixture evaluation currently supports batch_size=1.")

    model_size = normalize_model_size(model_size)
    set_seed(0)
    ood = as_bool(ood)
    best_selection = as_bool(best_selection)
    mixture_softmax = as_bool(mixture_softmax)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_data = load_eval_dataset(data_path, split=benchmark_split, max_examples=max_examples)

    base_model, tokenizer = load_base_model(model_size)
    base_model.to(device)
    base_model.eval()

    init_vector_db(
        config_path=config_path,
        base_model=base_model,
        tokenizer=tokenizer,
        model_size=model_size,
        prototype_num_samples=prototype_num_samples,
    )

    results = []
    with torch.no_grad():
        for idx in tqdm(range(len(eval_data)), desc="Evaluating", unit="item"):
            query_text = eval_data["inputs"][idx]
            task_name = eval_data["task"][idx]
            exclude_item = build_adapter_name(task_name, model_size) if ood else None

            layer_hits = perform_search_dw_surgery(
                query_text=query_text,
                base_model=base_model,
                tokenizer=tokenizer,
                layer_top_k=lora_num,
                coarse_top_m=activation_coarse_top_m,
                exclude_item=exclude_item,
            )
            if best_selection:
                layer_hits = build_best_selection_layer_hits(layer_hits, task_name, model_size)

            prediction, mixture_lora_paths = generate_mixture(
                base_model=base_model,
                tokenizer=tokenizer,
                prompt_text=eval_data["full_prompt"][idx],
                layer_hits=layer_hits,
                max_new_tokens=max_new_tokens,
                use_softmax=mixture_softmax,
                softmax_temperature=mixture_temperature,
            )

            sample = {
                "inputs": query_text,
                "targets": eval_data["targets"][idx],
                "metric": eval_data["metric"][idx],
                "domain": eval_data["domain"][idx],
                "task": task_name,
                "retrieved_adaptor": mixture_lora_paths,
                "predicted_answer": prediction,
                "outputs": prediction,
                "layer_hits": layer_hits,
                "prototype_num_samples": prototype_num_samples,
                "ood": ood,
                "best_selection": best_selection,
                "mixture_softmax": mixture_softmax,
                "mixture_temperature": mixture_temperature,
            }
            results.append(sample)
            print(f"generated_answer: {prediction}, expected_answer: {sample['targets']}")

    output_dir = os.path.dirname(res_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(res_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    import fire

    fire.Fire(eval_datasets)
