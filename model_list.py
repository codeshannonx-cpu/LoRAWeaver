"""Llama 2 model and LoRA adapter registry."""

SUPPORTED_MODEL_SIZES = {"7b", "13b"}
MODEL_SIZE_ALIASES = {
    "7b": "7b",
    "13b": "13b",
    "llama2-7b": "7b",
    "llama2-13b": "13b",
    "llama-2-7b": "7b",
    "llama-2-13b": "13b",
}

HF_MODEL_PATH_DICT = {
    "7b": "meta-llama/Llama-2-7b-hf",
    "13b": "meta-llama/Llama-2-13b-hf",
}


def normalize_model_size(model_size):
    normalized = MODEL_SIZE_ALIASES.get(str(model_size).strip().lower())
    if normalized not in SUPPORTED_MODEL_SIZES:
        raise ValueError(
            "Unsupported model_size. This code supports only "
            "'llama2-7b'/'7b' and 'llama2-13b'/'13b'."
        )
    return normalized


def get_base_model_path(model_size):
    return HF_MODEL_PATH_DICT[normalize_model_size(model_size)]


def build_lora_adapter_ref(model_size, task_name):
    model_size = normalize_model_size(model_size)
    return f"Styxxxx/llama2_{model_size}_lora-{task_name}"
