from typing import Type

import torch
import transformers
from peft import LoraConfig, get_peft_model
from transformers import LlamaTokenizer

from models.configuration import LlamaConfig
from pruning.pruner import get_pruned_model


DEFAULT_PAD_TOKEN = "[PAD]"


def find_all_linear_names(model):
    linear_cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def build_layer_qats(args):
    if not args.qat:
        return {i: {"w": 32, "a": 32, "kv": 32} for i in range(args.layer_num)}

    layers_qats = {
        i: {"w": args.uniform_bits, "a": args.uniform_bits, "kv": args.uniform_bits}
        for i in range(args.layer_num)
    }
    special_layers = {2, args.layer_num - 3, args.layer_num - 2, args.layer_num - 1}
    for layer_idx in special_layers:
        if 0 <= layer_idx < args.layer_num:
            layers_qats[layer_idx] = {"w": args.w_bits, "a": args.a_bits, "kv": args.kv_bits}
    return layers_qats


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg


def get_accelerate_model(args, logger, model_cls: Type[transformers.PreTrainedModel]):
    layers_qats = build_layer_qats(args)
    logger.info(layers_qats)

    config = LlamaConfig.from_pretrained(args.model_name_or_path)
    config.use_cache = False
    model_kwargs = {
        "pretrained_model_name_or_path": args.model_name_or_path,
        "config": config,
        "low_cpu_mem_usage": True,
        "layer_qats": layers_qats,
        "cache_dir": args.cache_dir,
        "device_map": "auto",
    }
    if args.qat:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = model_cls.from_pretrained(**model_kwargs)

    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path,
        cache_dir=args.cache_dir,
    )

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama" in args.model_name_or_path.lower() or isinstance(tokenizer, LlamaTokenizer):
        logger.info("Adding LLaMA special tokens.")
        tokenizer.add_special_tokens({
            "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
            "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
            "unk_token": tokenizer.convert_ids_to_tokens(
                model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
            ),
        })

    if args.pruning:
        logger.info("*******************BEGIN: Pruning Models*******************")
        model = get_pruned_model(model, tokenizer, args, logger)
        logger.info("*******************END: Pruning Models*******************")

    logger.info("*******************Adding LoRA Modules*******************")
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=find_all_linear_names(model),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    return model, tokenizer
