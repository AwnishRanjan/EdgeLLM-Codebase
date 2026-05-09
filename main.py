from utils import bnb_wrappers

import os
from os.path import join
import numpy as np
from tqdm import tqdm
import logging
from datasets import load_dataset
import evaluate

import torch
import transformers
from models.edge_llama_modelling import LlamaForCausalLM
from transformers import set_seed
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from utils.argument_parser import get_args
from utils.logger import get_logger
from utils.dataloader import make_data_module, get_wikitext2_dataset, get_ptb_dataset
from utils.layer_utils import get_base_model
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
    if args.do_mmlu_eval:
        if args.mmlu_dataset == 'mmlu-zs':
            mmlu_dataset = load_dataset("json", data_files={
                'eval': 'data/mmlu/zero_shot_mmlu_val.json',
                'test': 'data/mmlu/zero_shot_mmlu_test.json',
            })
            mmlu_dataset = mmlu_dataset.remove_columns('subject')
        elif args.mmlu_dataset == 'mmlu' or args.mmlu_dataset == 'mmlu-fs':
            mmlu_dataset = load_dataset("json", data_files={
                'eval': 'data/mmlu/five_shot_mmlu_val.json',
                'test': 'data/mmlu/five_shot_mmlu_test.json',
            })
        mmlu_dataset = mmlu_dataset[args.mmlu_split]
        if args.max_mmlu_samples is not None:
            mmlu_dataset = mmlu_dataset.select(range(args.max_mmlu_samples))
        abcd_idx = [
            tokenizer("A", add_special_tokens=False).input_ids[0],
            tokenizer("B", add_special_tokens=False).input_ids[0],
            tokenizer("C", add_special_tokens=False).input_ids[0],
            tokenizer("D", add_special_tokens=False).input_ids[0],
        ]
    if args.do_wikitext2_ppl:
        for ppl_dataset in args.ppl_dataset.split(','):
            if 'wikitext2' in ppl_dataset:
                _, wikitext2_test_dataset = get_wikitext2_dataset(tokenizer=tokenizer, batch_size=1)
            if 'ptb' in ppl_dataset:
                _, ptb_test_dataset = get_ptb_dataset(tokenizer=tokenizer, batch_size=1)

    accuracy = evaluate.load("accuracy")
    class MMLUEvalCallback(transformers.TrainerCallback):
        def on_evaluate(self, args, state, control, model, **kwargs):
            if args.do_mmlu_eval:
                data_loader = trainer.get_eval_dataloader(mmlu_dataset)
                source_max_len = trainer.data_collator.source_max_len
                trainer.data_collator.source_max_len = args.mmlu_source_max_len
                trainer.model.eval()
                preds, refs = [], []
                loss_mmlu = 0
                exit_layer_indices = trainer.eval_exit_layers
                preds_by_exit = [[] for _ in exit_layer_indices]

                linear_layers = []
                base_model = get_base_model(model)
                base_model.eval()
                
                for layer_idx in exit_layer_indices:
                    linear_layers.append(getattr(base_model.layers[layer_idx], f'linear_layer_{layer_idx}'))
                    
                with torch.no_grad():
                    for batch in tqdm(data_loader, total=len(data_loader)):
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        (loss, orig_logits, labels, hidden_states) = trainer.prediction_step_with_hidden_states(trainer.model,batch,prediction_loss_only=False)
                        exit_layers_logits = list()
                        for i, layer_idx in enumerate(exit_layer_indices):
                            hidden_state_idx = layer_idx + 1
                            exit_layers_logits.append(torch.nn.functional.softmax(linear_layers[i](hidden_states[hidden_state_idx].to(trainer.model.lm_head.weight.dtype).to(linear_layers[i].weight.device)), dim=-1).to('cpu'))
                        
                        logits = torch.stack(exit_layers_logits, dim=0).to('cpu')
                        topk = torch.topk(logits, k=1, dim=0)[0].squeeze(dim=0)
                        final_logits = topk/torch.sum(topk, dim=2)[:,:,None]

                        exit_layers_preds = list() 
                        for logit in exit_layers_logits:
                            label_non_zero_id = (batch['labels'][0] != -100).nonzero()[0][0]
                            logit_abcd = logit[0][label_non_zero_id-1][abcd_idx]
                            exit_layers_preds.append(torch.argmax(logit_abcd).item())

                        for exit_idx, exit_pred in enumerate(exit_layers_preds):
                            preds_by_exit[exit_idx].append(exit_pred)

                        for i, logit in enumerate(final_logits):
                            label_non_zero_id = (batch['labels'][i] != -100).nonzero()[0][0]
                            logit_abcd = logit[label_non_zero_id-1][abcd_idx]
                            preds.append(torch.argmax(logit_abcd).item())
            
                        labels = labels[labels != IGNORE_INDEX].view(-1, 2)[:,0]
                        refs += [abcd_idx.index(label) for label in labels.tolist()]
                        loss_mmlu += loss.item()

                results = {'mmlu_loss':loss_mmlu/len(data_loader)}
                subject = mmlu_dataset['subject']
                subjects = {}
                for s in set(subject):
                    subjects[s] = {'refs': [], 'preds_comb': []}
                    for exit_idx in range(len(exit_layer_indices)):
                        subjects[s][f'preds_exitlayer{exit_idx + 1}'] = []

                for item_idx, s in enumerate(subject):
                    subjects[s]['refs'].append(refs[item_idx])
                    for exit_idx, exit_preds in enumerate(preds_by_exit):
                        subjects[s][f'preds_exitlayer{exit_idx + 1}'].append(exit_preds[item_idx])
                    subjects[s]['preds_comb'].append(preds[item_idx])

                subject_scores_by_exit = [[] for _ in exit_layer_indices]
                subject_scores_comb = []

                for subject in subjects:
                    for exit_idx in range(len(exit_layer_indices)):
                        subject_score = accuracy.compute(
                            references=subjects[subject]['refs'],
                            predictions=subjects[subject][f'preds_exitlayer{exit_idx + 1}']
                        )['accuracy']
                        subject_scores_by_exit[exit_idx].append(subject_score)
                    subject_score_comb = accuracy.compute(
                        references=subjects[subject]['refs'],
                        predictions=subjects[subject]['preds_comb']
                    )['accuracy']

                    subject_scores_comb.append(subject_score_comb)

                for exit_idx, subject_scores in enumerate(subject_scores_by_exit):
                    results[f'mmlu_{args.mmlu_split}_accuracy_exitlayer{exit_idx + 1}'] = np.mean(subject_scores)
                results[f'mmlu_{args.mmlu_split}_accuracy_comb'] = np.mean(subject_scores_comb)

                logger.info(", ".join(
                    [str(np.mean(subject_scores)) for subject_scores in subject_scores_by_exit]
                    + [str(np.mean(subject_scores_comb))]
                ))

                trainer.log(results)
                trainer.data_collator.source_max_len = source_max_len
                trainer.model.train()
            elif args.do_wikitext2_ppl:
                wikitext2_dataloader = trainer.get_eval_dataloader(wikitext2_test_dataset)
                trainer.data_collator.dataset_name = "ppl"
                trainer.model.eval()
                wiki_loss_container, ptb_loss_container = [], []
                with torch.no_grad():
                    for batch in tqdm(wikitext2_dataloader, total=len(wikitext2_dataloader)):
                        loss, logits, labels, hidden_states = trainer.prediction_step_with_hidden_states(trainer.model, batch, prediction_loss_only=False,)
                        logits = logits[0]

                        shift_logit = logits[:, :-1, :].contiguous()
                        shift_label = batch['labels'][:, 1:].contiguous()
                        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                        wiki_ppl_loss = loss_fct(shift_logit.reshape(-1, shift_logit.size(-1)), shift_label.view(-1))
                        wiki_loss_container.append(wiki_ppl_loss)
                    
                wiki_ppl = np.exp(torch.cat(wiki_loss_container, dim=-1).mean().item()).item()
                wiki_results = {'wikitext2_perplexity': wiki_ppl}
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
