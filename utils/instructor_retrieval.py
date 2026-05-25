import json
import os
from collections import defaultdict
from functools import lru_cache

import numpy as np
import torch
from huggingface_hub import snapshot_download
from peft import PeftConfig
from peft.utils.constants import CONFIG_NAME
from tqdm import tqdm

from model_list import (
    build_lora_adapter_ref,
    normalize_model_size,
)


LAYERWISE_INDEX_MODE_ACTIVATION = "activation_delta"

global_layerwise_index = None
global_layerwise_index_mode = None
global_lora_factor_cache = {}


@lru_cache(maxsize=None)
def _resolve_adapter_dir(lora_path):
    def has_adapter_config(path):
        return os.path.isfile(os.path.join(path, CONFIG_NAME))

    if os.path.isdir(lora_path):
        return lora_path

    try:
        return snapshot_download(repo_id=lora_path, local_files_only=True)
    except Exception:
        return snapshot_download(repo_id=lora_path)


def build_adapter_name(model_name, model_size):
    return build_lora_adapter_ref(model_size, model_name)


def initialize_index_dw(
    models,
    model_size="7b",
    cache_dir="cache/layerwise_activation",
    force_rebuild=True,
    base_model=None,
    tokenizer=None,
    activation_batch_size=4,
    activation_max_length=512,
    prototype_num_samples=20,
):
    global global_layerwise_index, global_layerwise_index_mode

    model_size = normalize_model_size(model_size)
    prototype_num_samples = int(prototype_num_samples)
    if prototype_num_samples < 1:
        raise ValueError("prototype_num_samples must be >= 1.")

    cache_subdir = os.path.join(
        cache_dir,
        f"modelsize_{model_size}_activation_protosamples_{prototype_num_samples}",
    )

    if not force_rebuild and layerwise_index_exists(cache_subdir):
        global_layerwise_index, global_layerwise_index_mode = load_layerwise_activation_index(cache_subdir)
        return

    if base_model is None or tokenizer is None:
        raise ValueError("initialize_index_dw() requires base_model and tokenizer.")

    global_layerwise_index = build_activation_layerwise_lora_index(
        models=models,
        base_model=base_model,
        tokenizer=tokenizer,
        model_size=model_size,
        batch_size=activation_batch_size,
        max_length=activation_max_length,
        prototype_num_samples=prototype_num_samples,
    )
    global_layerwise_index_mode = LAYERWISE_INDEX_MODE_ACTIVATION
    save_layerwise_activation_index(global_layerwise_index, cache_subdir)


def _candidate_layer_mapping_keys(module_key):
    if not module_key:
        return []

    keys = []

    def add_key(key):
        if key and key not in keys:
            keys.append(key)

    add_key(module_key)

    stripped = module_key
    while stripped.startswith("base_model.model."):
        stripped = stripped[len("base_model.model.") :]
        add_key(stripped)
    while stripped.startswith("model."):
        stripped = stripped[len("model.") :]
        add_key(stripped)

    for key in list(keys):
        if not key.startswith("model."):
            add_key(f"model.{key}")
        if not key.startswith("base_model.model."):
            add_key(f"base_model.model.{key}")

    return keys


def _resolve_named_module_map(base_model, target_layers):
    named_modules = dict(base_model.named_modules())
    resolved = {}
    for target_layer in target_layers:
        for candidate_key in _candidate_layer_mapping_keys(target_layer):
            if candidate_key in named_modules:
                resolved[target_layer] = candidate_key
                break
    return resolved, named_modules


def _masked_mean_pool_hidden(hidden_states, attention_mask=None):
    if hidden_states.ndim == 2:
        return hidden_states.float()

    hidden_states = hidden_states.float()
    if attention_mask is None:
        return hidden_states.mean(dim=1)

    mask = attention_mask[:, : hidden_states.shape[1]].to(
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    mask = mask.unsqueeze(-1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (hidden_states * mask).sum(dim=1) / denom


def _capture_mean_pooled_layer_inputs(
    base_model,
    tokenizer,
    text_list,
    target_layers,
    batch_size=4,
    max_length=512,
):
    if not text_list:
        return {}

    device = next(base_model.parameters()).device
    resolved_layers, named_modules = _resolve_named_module_map(base_model, target_layers)
    pooled_by_layer = defaultdict(list)

    for start in range(0, len(text_list), batch_size):
        batch_texts = text_list[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            padding=True,
        ).to(device)
        attention_mask = encoded.get("attention_mask", None)
        batch_pooled = defaultdict(list)
        handles = []

        for canonical_name, actual_name in resolved_layers.items():
            module = named_modules[actual_name]

            def make_hook(layer_name):
                def hook(_module, hook_inputs, _hook_output):
                    if not hook_inputs:
                        return
                    hidden_states = hook_inputs[0]
                    if not torch.is_tensor(hidden_states):
                        return
                    pooled = _masked_mean_pool_hidden(
                        hidden_states.detach(),
                        attention_mask=attention_mask,
                    )
                    batch_pooled[layer_name].append(pooled.cpu())

                return hook

            handles.append(module.register_forward_hook(make_hook(canonical_name)))

        try:
            with torch.no_grad():
                base_model(
                    input_ids=encoded["input_ids"],
                    attention_mask=attention_mask,
                    use_cache=False,
                )
        finally:
            for handle in handles:
                handle.remove()

        for layer_name, chunks in batch_pooled.items():
            if chunks:
                pooled_by_layer[layer_name].append(torch.cat(chunks, dim=0))

    mean_inputs = {}
    for layer_name, pooled_chunks in pooled_by_layer.items():
        if pooled_chunks:
            stacked = torch.cat(pooled_chunks, dim=0)
            mean_inputs[layer_name] = stacked.mean(dim=0).numpy().astype(np.float32)

    return mean_inputs


def _resolve_adapter_file(lora_path, filename):
    local_dir = _resolve_adapter_dir(lora_path)
    local_file = os.path.join(local_dir, filename)
    if os.path.exists(local_file):
        return local_file
    return None


def _load_lora_state(lora_path):
    import safetensors.torch as sf

    weights_file = _resolve_adapter_file(lora_path, "adapter_model.safetensors")
    if weights_file is not None:
        return sf.load_file(weights_file)

    weights_file = _resolve_adapter_file(lora_path, "adapter_model.bin")
    if weights_file is None:
        raise FileNotFoundError(f"No adapter_model.safetensors/.bin for {lora_path}")
    return torch.load(weights_file, map_location="cpu")


def _load_lora_config(lora_path):
    return PeftConfig.from_pretrained(_resolve_adapter_dir(lora_path), local_files_only=True)


def _resolve_layer_pattern_value(layer_name, pattern_dict, default_value):
    if not pattern_dict:
        return default_value
    for key in _candidate_layer_mapping_keys(layer_name):
        if key in pattern_dict:
            return pattern_dict[key]
    return default_value


def _extract_layer_lora_factors(lora_path):
    state_dict = _load_lora_state(lora_path)
    cfg = _load_lora_config(lora_path)

    rank_pattern = getattr(cfg, "rank_pattern", None) or {}
    alpha_pattern = getattr(cfg, "alpha_pattern", None) or {}
    default_r = float(cfg.r)
    default_alpha = float(cfg.lora_alpha)

    grouped = {}
    for key, val in state_dict.items():
        if "lora_A.weight" in key:
            layer = key.replace(".lora_A.weight", "").replace("base_model.model.", "")
            grouped.setdefault(layer, {})["A"] = val.float().cpu().contiguous()
        elif "lora_B.weight" in key:
            layer = key.replace(".lora_B.weight", "").replace("base_model.model.", "")
            grouped.setdefault(layer, {})["B"] = val.float().cpu().contiguous()

    factors = {}
    for layer, parts in grouped.items():
        if "A" not in parts or "B" not in parts:
            continue

        layer_r = float(_resolve_layer_pattern_value(layer, rank_pattern, default_r))
        layer_alpha = float(_resolve_layer_pattern_value(layer, alpha_pattern, default_alpha))
        if layer_r <= 0:
            continue

        factors[layer] = {
            "A": parts["A"],
            "B": parts["B"],
            "scaling": layer_alpha / layer_r,
        }

    return factors


def _get_layer_lora_factors_cached(lora_path):
    cached = global_lora_factor_cache.get(lora_path)
    if cached is None:
        cached = _extract_layer_lora_factors(lora_path)
        global_lora_factor_cache[lora_path] = cached
    return cached


def _resolve_adapter_layer_factors(adapter_factors, layer_name):
    if adapter_factors is None:
        return None
    if layer_name in adapter_factors:
        return adapter_factors[layer_name]
    for key in _candidate_layer_mapping_keys(layer_name):
        if key in adapter_factors:
            return adapter_factors[key]
    return None


def _project_pooled_hidden_with_lora(pooled_hidden, layer_factors):
    if pooled_hidden is None or layer_factors is None:
        return None

    A = layer_factors["A"].float()
    B = layer_factors["B"].float()
    scaling = float(layer_factors.get("scaling", 1.0))
    pooled_hidden = torch.as_tensor(pooled_hidden, dtype=torch.float32)

    if (
        pooled_hidden.ndim != 1
        or A.ndim != 2
        or B.ndim != 2
        or pooled_hidden.shape[0] != A.shape[1]
    ):
        return None

    projected = pooled_hidden.unsqueeze(0) @ A.transpose(0, 1) @ B.transpose(0, 1)
    return (projected.squeeze(0) * scaling).cpu().numpy().astype(np.float32)


def _normalize_np_vector(vector):
    if vector is None:
        return None
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return None
    return vector / norm


def _iter_layerwise_buckets(layerwise_index):
    for layer_name, bucket in layerwise_index.items():
        if isinstance(bucket, dict):
            yield layer_name, bucket


def _all_layerwise_adapter_paths(layerwise_index):
    adapter_paths = []
    seen = set()
    for _, bucket in _iter_layerwise_buckets(layerwise_index):
        for lora_path in bucket.get("lora_paths", []):
            if lora_path not in seen:
                seen.add(lora_path)
                adapter_paths.append(lora_path)
    return adapter_paths


def _finalize_activation_layerwise_index(layerwise_index):
    for _, bucket in _iter_layerwise_buckets(layerwise_index):
        matrix = np.asarray(bucket.get("matrix", []), dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        bucket["matrix"] = matrix
        bucket["path_to_index"] = {
            lora_path: idx for idx, lora_path in enumerate(bucket.get("lora_paths", []))
        }
    return layerwise_index


def build_activation_layerwise_lora_index(
    models,
    base_model,
    tokenizer,
    model_size="7b",
    batch_size=4,
    max_length=512,
    prototype_num_samples=20,
):
    prototype_num_samples = int(prototype_num_samples)
    if prototype_num_samples < 1:
        raise ValueError("prototype_num_samples must be >= 1.")

    layerwise_buckets = {}
    for model in tqdm(
        models,
        desc=f"Building activation-conditioned layer index ({prototype_num_samples} samples)",
        unit="adapter",
    ):
        lora_path = build_adapter_name(model["model_name"], model_size)
        adapter_factors = _get_layer_lora_factors_cached(lora_path)
        if not adapter_factors:
            continue

        sample_texts = [
            sample.get("inputs", "")
            for sample in model.get("sample", [])[:prototype_num_samples]
            if sample.get("inputs")
        ]
        if not sample_texts:
            continue

        pooled_inputs = _capture_mean_pooled_layer_inputs(
            base_model=base_model,
            tokenizer=tokenizer,
            text_list=sample_texts,
            target_layers=list(adapter_factors.keys()),
            batch_size=batch_size,
            max_length=max_length,
        )

        for layer_name, layer_factors in adapter_factors.items():
            prototype = _project_pooled_hidden_with_lora(
                pooled_hidden=pooled_inputs.get(layer_name),
                layer_factors=layer_factors,
            )
            prototype = _normalize_np_vector(prototype)
            if prototype is None:
                continue

            layerwise_buckets.setdefault(layer_name, []).append(
                {
                    "lora_path": lora_path,
                    "prototype": prototype,
                }
            )

    layerwise_index = {}
    for layer_name, items in layerwise_buckets.items():
        matrix = np.stack([item["prototype"] for item in items], axis=0).astype(np.float32)
        layerwise_index[layer_name] = {
            "lora_paths": [item["lora_path"] for item in items],
            "matrix": matrix,
        }

    print(f"Built activation-conditioned layer-wise index for {len(layerwise_index)} layers")
    return _finalize_activation_layerwise_index(layerwise_index)


def save_layerwise_activation_index(layerwise_index, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    manifest = {
        "version": 2,
        "mode": LAYERWISE_INDEX_MODE_ACTIVATION,
        "layers": [],
    }

    for layer_name, bucket in _iter_layerwise_buckets(layerwise_index):
        safe_layer_name = layer_name.replace("/", "__").replace(".", "_")
        matrix_file = f"{safe_layer_name}.npy"
        np.save(os.path.join(save_dir, matrix_file), np.asarray(bucket["matrix"], dtype=np.float32))
        manifest["layers"].append(
            {
                "layer_name": layer_name,
                "matrix_file": matrix_file,
                "lora_paths": list(bucket.get("lora_paths", [])),
            }
        )

    with open(os.path.join(save_dir, "manifest.json"), "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print(f"Saved activation-conditioned layer-wise index to {save_dir}")


def load_layerwise_activation_index(save_dir):
    manifest_path = os.path.join(save_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"No manifest found at {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest = json.load(file)

    if manifest.get("mode") != LAYERWISE_INDEX_MODE_ACTIVATION:
        raise ValueError(
            f"Unsupported layerwise index mode in {manifest_path}: {manifest.get('mode')}"
        )

    layerwise_index = {}
    for layer_info in manifest.get("layers", []):
        layer_name = layer_info["layer_name"]
        matrix = np.load(os.path.join(save_dir, layer_info["matrix_file"])).astype(np.float32)
        layerwise_index[layer_name] = {
            "lora_paths": list(layer_info.get("lora_paths", [])),
            "matrix": matrix,
        }

    print(f"Loaded activation-conditioned layer-wise index from {save_dir}")
    return _finalize_activation_layerwise_index(layerwise_index), manifest.get("mode")


def layerwise_index_exists(save_dir):
    return os.path.exists(os.path.join(save_dir, "manifest.json"))


def retrieve_topk_per_layer_activation(
    query_text,
    layerwise_index,
    base_model,
    tokenizer,
    layer_k=3,
    exclude_item=None,
    max_length=512,
):
    if base_model is None or tokenizer is None:
        raise ValueError("retrieve_topk_per_layer_activation() requires base_model and tokenizer.")

    target_layers = [layer_name for layer_name, _ in _iter_layerwise_buckets(layerwise_index)]
    query_inputs = _capture_mean_pooled_layer_inputs(
        base_model=base_model,
        tokenizer=tokenizer,
        text_list=[query_text],
        target_layers=target_layers,
        batch_size=1,
        max_length=max_length,
    )

    candidate_adapters = [
        adapter for adapter in _all_layerwise_adapter_paths(layerwise_index) if adapter != exclude_item
    ]
    candidate_factor_cache = {
        lora_path: _get_layer_lora_factors_cached(lora_path) for lora_path in candidate_adapters
    }

    hits = {}
    for layer_name, bucket in _iter_layerwise_buckets(layerwise_index):
        pooled_hidden = query_inputs.get(layer_name)
        if pooled_hidden is None:
            hits[layer_name] = []
            continue

        layer_hits = []
        for lora_path in bucket.get("lora_paths", []):
            if lora_path == exclude_item:
                continue

            row_idx = bucket.get("path_to_index", {}).get(lora_path)
            if row_idx is None:
                continue

            layer_factors = _resolve_adapter_layer_factors(
                candidate_factor_cache.get(lora_path),
                layer_name,
            )
            query_delta = _project_pooled_hidden_with_lora(
                pooled_hidden=pooled_hidden,
                layer_factors=layer_factors,
            )
            query_delta = _normalize_np_vector(query_delta)
            if query_delta is None:
                continue

            prototype = bucket["matrix"][int(row_idx)]
            layer_hits.append(
                {
                    "lora_path": lora_path,
                    "score": float(np.dot(query_delta, prototype)),
                }
            )

        layer_hits.sort(key=lambda item: item["score"], reverse=True)
        hits[layer_name] = layer_hits[:layer_k]

    return hits


def perform_search_dw_surgery(
    query_text,
    base_model,
    tokenizer,
    layer_top_k=3,
    coarse_top_m=8,
    exclude_item=None,
):
    if global_layerwise_index is None:
        raise ValueError("Layerwise index has not been initialized. Call initialize_index_dw() first.")

    return retrieve_topk_per_layer_activation(
        query_text=query_text,
        layerwise_index=global_layerwise_index,
        layer_k=layer_top_k,
        exclude_item=exclude_item,
        base_model=base_model,
        tokenizer=tokenizer,
    )


def _scores_to_softmax_weights(score_by_adapter, chosen_adapters, temperature=1.0):
    if not chosen_adapters:
        return []

    temperature = max(float(temperature), 1e-6)
    scores = np.asarray(
        [float(score_by_adapter.get(adapter, 0.0)) for adapter in chosen_adapters],
        dtype=np.float32,
    )
    logits = scores * temperature
    logits = logits - float(np.max(logits))
    exp_scores = np.exp(logits)
    total = float(exp_scores.sum())
    if total <= 1e-12:
        return [1.0 / len(chosen_adapters)] * len(chosen_adapters)
    return (exp_scores / total).astype(np.float32).tolist()


def _scores_to_normalized_weights(score_by_adapter, chosen_adapters):
    if not chosen_adapters:
        return []

    scores = np.asarray(
        [max(float(score_by_adapter.get(adapter, 0.0)), 0.0) for adapter in chosen_adapters],
        dtype=np.float32,
    )
    total = float(scores.sum())
    if total <= 1e-12:
        return [1.0 / len(chosen_adapters)] * len(chosen_adapters)
    return (scores / total).astype(np.float32).tolist()


def _normalize_layer_hits_to_weights(layer_hits_for_module, use_softmax=False, softmax_temperature=1.0):
    if not layer_hits_for_module:
        return [], []

    score_by_adapter = defaultdict(float)
    for hit in layer_hits_for_module:
        lora_path = hit["lora_path"]
        raw_score = float(hit.get("score", 0.0))
        score = raw_score if use_softmax else max(raw_score, 0.0)
        score_by_adapter[lora_path] += score

    adapters = list(score_by_adapter.keys())
    if use_softmax:
        weights = _scores_to_softmax_weights(
            score_by_adapter,
            adapters,
            temperature=softmax_temperature,
        )
    else:
        weights = _scores_to_normalized_weights(score_by_adapter, adapters)
    return adapters, weights


def collect_layerwise_lora_paths(layer_hits):
    lora_paths = []
    seen = set()
    for hits_for_layer in layer_hits.values():
        for hit in hits_for_layer:
            lora_path = hit.get("lora_path")
            if lora_path is None or lora_path in seen:
                continue
            seen.add(lora_path)
            lora_paths.append(lora_path)
    return lora_paths


def build_layerwise_direct_lora_mapping(
    layer_hits,
    lora_paths,
    batch_size=1,
    use_softmax=False,
    softmax_temperature=1.0,
):
    path_to_index = {lora_path: idx for idx, lora_path in enumerate(lora_paths)}
    default_mapping = torch.zeros((batch_size, len(lora_paths)), dtype=torch.float32)
    mapping = {"__default__": default_mapping}

    for layer_name, hits_for_layer in layer_hits.items():
        selected_paths, weights = _normalize_layer_hits_to_weights(
            hits_for_layer,
            use_softmax=use_softmax,
            softmax_temperature=softmax_temperature,
        )
        layer_mapping = torch.zeros((batch_size, len(lora_paths)), dtype=torch.float32)
        for lora_path, weight in zip(selected_paths, weights):
            adapter_idx = path_to_index.get(lora_path)
            if adapter_idx is not None:
                layer_mapping[:, adapter_idx] = float(weight)
        mapping[layer_name] = layer_mapping

    return mapping
