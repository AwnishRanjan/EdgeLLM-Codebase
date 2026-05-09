import os
from os.path import join
import numpy as np
from tqdm import tqdm
import logging
import torch
import transformers
from models.quantized_llama_modelling import LlamaForCausalLM
from transformers import set_seed
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from utils.argument_parser import get_args
from utils.logger import get_logger
from utils.dataloader import make_data_module, get_wikitext2_dataset, get_ptb_dataset
from utils.model_loader import get_accelerate_model
from utils.trainer_wrappers import EdgeLLMTrainer

if torch.cuda.is_available():   
    torch.backends.cuda.matmul.allow_tf32 = True
logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)

def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.bits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )
def train():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:26"
    args, training_args = get_args()
    logger = get_logger("Edge_LLM", args.log_dir)
    set_seed(args.seed)
    logger.info(args)

    logger.info("*****************BEGIN:loading model***************")
    model, tokenizer = get_accelerate_model(args, logger, LlamaForCausalLM)
    logger.info("*****************END:loading model***************")

    logger.info("*****************BEGIN:loading dataset***************")
    data_module = make_data_module(tokenizer=tokenizer, args=args)
    logger.info("*****************END:loading dataset***************")
    
    trainer = EdgeLLMTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
    )

    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)
    if args.do_wikitext2_ppl:
        for ppl_dataset in args.ppl_dataset.split(','):
            if 'wikitext2' in ppl_dataset:
                _, wikitext2_test_dataset = get_wikitext2_dataset(tokenizer=tokenizer, batch_size=1)
            if 'ptb' in ppl_dataset:
                _, ptb_test_dataset = get_ptb_dataset(tokenizer=tokenizer, batch_size=1)

    class MMLUEvalCallback(transformers.TrainerCallback):
        def on_evaluate(self, args, state, control, model, **kwargs):
            if args.do_wikitext2_ppl:
                wikitext2_dataloader = trainer.get_eval_dataloader(wikitext2_test_dataset)
                trainer.data_collator.dataset_name = "ppl"
                trainer.model.eval()
                wiki_loss_container, ptb_loss_container = [], []
                with torch.no_grad():
                    for batch in tqdm(wikitext2_dataloader, total=len(wikitext2_dataloader)):
                        loss, logits, labels = trainer.prediction_step(trainer.model, batch, prediction_loss_only=False,)
                        logits = logits[0]

                        shift_logit = logits[:, :-1, :].contiguous()
                        shift_label = batch['labels'][:, 1:].contiguous()
                        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                        wiki_ppl_loss = loss_fct(shift_logit.reshape(-1, shift_logit.size(-1)), shift_label.view(-1))
                        wiki_loss_container.append(wiki_ppl_loss)
                    
                wiki_ppl = np.exp(torch.cat(wiki_loss_container, dim=-1).mean().item()).item()
                wiki_results = {'wikitest2_perplexity': wiki_ppl}
                logger.info(wiki_results)
                trainer.data_collator.dataset_name = 'alpaca'
                trainer.model.train()
    trainer.add_callback(MMLUEvalCallback)

    print_trainable_parameters(args, model)
    if args.do_train:
        logger.info("*** Train ***")
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    train()
