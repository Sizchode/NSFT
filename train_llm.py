import argparse
import logging
from datetime import datetime

import torch
import wandb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.optim import AdamW
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from trl import DataCollatorForCompletionOnlyLM, SFTConfig
import os
from META import LLM_DATASET_PATHS as dataset_paths  # Assuming this exists
from diet.dataset import MQUAKE, CLUTRR, TQA
from diet.dataset.genloader_2 import BoolQ, RTE, HellaSwag, WinoGrande, ARC, OBQA, LogiQA2
from diet.mlps.ns_linear2 import NeuroselectiveLinear
from diet.ns_transformer5 import NeuroselectiveTransformer5
from diet.tracker6 import NeuronTracker6 as NeuronTracker
from peft import get_peft_model, LoraConfig, AdaLoraConfig, LoHaConfig, LoKrConfig
from diet.trainers import CustomSFTTrainerV2
from diet.ns_lora import NSLoraLinear  # Import the NSLoraLinear class
from trainer_utils import (
    set_seed,
    setup_lora,
    calculate_jaccard_similarity,
    calculate_directed_coverage
)
import sys
from transformers import Qwen2ForCausalLM 
import torch.distributed as dist
from torch_flops import (
    flops_forward, flops_train_step, extract_inputs_from_batch,
    TotalsSpec, aggregate_totals_flops, summarize_totals,
    estimate_flops_infer, estimate_flops_train,
)
import torch.nn as nn
import gc


logger = logging.getLogger(__name__)


# ===== Debug helpers for LLM dataloaders =====
from typing import Any, Dict, Iterable, List, Tuple, Union, Optional
import torch
from torch.utils.data import DataLoader

def peek_dataloader(dl: DataLoader, tokenizer=None, n_batches: int = 2, prefix: str = "[DEBUG]"):
    """
    get tokens num
    """
    print(f"{prefix} dl.type = {type(dl)}  dataset.type = {type(getattr(dl, 'dataset', None))}")
    try:
        print(f"{prefix} dataset.len = {len(dl.dataset)}  batch_size = {getattr(dl, 'batch_size', None)}")
    except Exception:
        pass
    cf = getattr(dl, "collate_fn", None)
    print(f"{prefix} collate_fn = {getattr(cf, '__name__', str(cf))}")

    D_partial = 0
    pad_id = getattr(getattr(tokenizer, 'pad_token_id', None), '__int__', lambda: None)() if tokenizer else None

    for i, batch in enumerate(dl):
        if i >= n_batches:
            break
        print(f"{prefix} ----- batch {i} -----")
        print(f"{prefix} batch.type = {type(batch)}")

        if isinstance(batch, dict):
            print(f"{prefix} dict.keys = {list(batch.keys())}")
            for k, v in batch.items():
                if torch.is_tensor(v):
                    print(f"{prefix}  {k}: shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device}")
                elif isinstance(v, list):
                    print(f"{prefix}  {k}: list(len={len(v)})  sample0={str(v[0])[:80]}...")
            # 计算本 batch tokens
            if "attention_mask" in batch and torch.is_tensor(batch["attention_mask"]):
                tok = int(batch["attention_mask"].long().sum().item())
            elif "input_ids" in batch and torch.is_tensor(batch["input_ids"]):
                ids = batch["input_ids"]
                if pad_id is not None:
                    tok = int((ids != pad_id).long().sum().item())
                else:
                    tok = int(ids.numel())
            elif "text" in batch and isinstance(batch["text"], list) and tokenizer is not None:
                enc = tokenizer(batch["text"], add_special_tokens=True, padding=False, truncation=False)
                tok = sum(len(x) for x in enc["input_ids"])
            else:
                tok = 0
            D_partial += tok
            print(f"{prefix} batch_tokens = {tok}")

        elif isinstance(batch, (list, tuple)):
            if len(batch) > 0 and isinstance(batch[0], str) and tokenizer is not None:
                # list[str]
                enc = tokenizer(list(batch), add_special_tokens=True, padding=False, truncation=False)
                tok = sum(len(x) for x in enc["input_ids"])
                D_partial += tok
                print(f"{prefix} list[str] size={len(batch)}  batch_tokens={tok}  sample0={batch[0][:80]}...")
            elif len(batch) > 0 and torch.is_tensor(batch[0]):
                # (input_ids, attention_mask, ...)
                ids = batch[0]
                print(f"{prefix} tuple[0] tensor shape={tuple(ids.shape)} dtype={ids.dtype} device={ids.device}")
                if len(batch) > 1 and torch.is_tensor(batch[1]):
                    tok = int(batch[1].long().sum().item())
                else:
                    if pad_id is not None:
                        tok = int((ids != pad_id).long().sum().item())
                    else:
                        tok = int(ids.numel())
                D_partial += tok
                print(f"{prefix} batch_tokens = {tok}")
            else:
                print(f"{prefix} tuple/list (unsupported content preview)")

        else:
            print(f"{prefix} unknown batch type; preview: {str(batch)[:120]}")

    print(f"{prefix} D_partial (first {n_batches} batches) = {D_partial}")
    return D_partial





def main():
    parser = argparse.ArgumentParser(description='Hidden Dimension Pruning Training for LLMs')

    parser.add_argument('--model', type=str, required=True, help='Model name or path')
    parser.add_argument('--task', "--dataset", type=str,
                        choices=['tqa', 'mquake', 'clutrr', 'gsm8k', 'musique', 'babi', 'cogs',
                                 'boolq', 'rte', 'hellaswag', 'winogrande', 'arc-e', 'arc-c', 'obqa', 'logiqa2'],
                        default="clutrr",
                        help='Task/dataset to evaluate on')
    parser.add_argument('--mode', type=str,
                        choices=["ns", "lora", "adalora", "loha", "lokr", "finetune", "baseline", "nslora","mag_pt", "mag_tp","wanda_p","transfer", "covplot","wanda_tp", "calibration"],
                        default="nslora",  # Changed default to nslora
                        help="Training mode: 'ns' for hidden dim pruning, 'nslora' for neuron-selective LoRA, etc.")
    parser.add_argument('--apply_chat_template',
                        action='store_true',
                        help='Using chat template in the prompt; False by default')

    parser.add_argument('--lr', "--learning_rate", type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--recovery_lr', "--recovery)learning_rate", type=float, default=1e-5, help='recovery Learning rate')

    parser.add_argument('--wd', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--train_batch', "--batch_size", type=int, default=8, help='Training batch size')
    parser.add_argument('--eval_batch', type=int, default=32, help='Evaluation batch size')
    parser.add_argument('--num_epoch', "--num_epochs", type=int, default=10, help='Number of epochs to fine-tune')
    parser.add_argument('--train_size', type=int, default=0, help='Number of training examples to use (0 for all)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of steps to accumulate gradients')


    parser.add_argument('--active_threshold', type=float, default=0.01,
                        help='Activation threshold for finding active neurons')
    parser.add_argument('--active_thresholds', type=float, nargs=2, default=None,
                        help='Two activation thresholds for neuron selection (e.g., 0.01 0.05)')
    parser.add_argument('--use_abs_threshold', action='store_true',
                        help='Use absolute threshold values for activation')
    parser.add_argument('--active_sample_ratio', type=float, default=0.1,
                        help='Sample ratio of training data for activation tracking')
    parser.add_argument("--topk_ratio",type=float, default=0.30, metavar="R",
                        help="Fraction (0,1] of neurons kept per score when selecting active neurons")

    parser.add_argument('--aggregation_mode', type=str, choices=['union', 'intersection'], default='union',
                        help='How to aggregate hidden dimension indices (union or intersection)')
    parser.add_argument('--hidden_dim_patterns', nargs='+', default=['gate_proj', 'up_proj', 'fc1'],
                        help='Patterns for layers determining hidden dim activity')
    parser.add_argument('--input_conn_patterns', nargs='+', default=['.down_proj', '.fc2', '.o_proj'],
                        help='Patterns for layers with hidden dim as input (use . prefix for specificity)')
    parser.add_argument('--output_conn_patterns', nargs='+',
                        default=['.up_proj', '.gate_proj', '.fc1', '.q_proj', '.k_proj', '.v_proj'],
                        help='Patterns for layers with hidden dim as output (use . prefix for specificity)')
    parser.add_argument('--tune_pruned', action='store_true', help='Make pruned layers trainable')

    parser.add_argument('--lora_r', type=int, default=16,
                        help='Rank of the LoRA updates')
    parser.add_argument('--lora_alpha', type=float, default=32,
                        help='Scaling factor for LoRA')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                        help='Dropout rate for LoRA layers')

    parser.add_argument('--output_dir', default='~/scratch/pyu12/outputs', type=str,  # Example path
                        help='Output directory for checkpoints and outputs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_wandb', action='store_true', help='Log metrics to Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='hidden-dim-pruning-llm', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity/username')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='WandB run name')
    parser.add_argument('--log_interval', type=int, default=10, help='Log to WandB every N steps')

    parser.add_argument('--tqa_fold_num', type=int, default=0, required=False,
                        help='The fold number to use for truthfulqa (0 or 1)')

    parser.add_argument("--schedule", action="store_true", help="enable linear schedule")
    parser.add_argument("--tune_attn", action="store_true", help="tune attn proj during tracking (if tracker supports)")
    parser.add_argument('--dev_mode', action='store_true',
                        help='Use a held-out dev set from training set for validation (grid search); do NOT use real test set')
    parser.add_argument( "--local_rank", type=int, default=-1, help="(DeepSpeed) local process rank — do not set manually"
    )
    parser.add_argument("--deepspeed", type=str, default=None, help="(DeepSpeed) path to deepspeed config json"
    )
    parser.add_argument('--use_deepspeed', action='store_true',
                    help='Use DeepSpeed (expects --deepspeed config path).')
    parser.add_argument('--source_task', type=str, default=None,
                        choices=['tqa','mquake','clutrr',
                                 'boolq','rte','hellaswag','winogrande','arc-e','arc-c','obqa','logiqa2'],
                        help='Donor task used to select/prune the subnetwork (transfer mode).')
    parser.add_argument('--source_ratio', type=float, default=None,
                        help='Top-k ratio (ρ) used on the SOURCE task during selection; falls back to --topk_ratio.')

    args = parser.parse_args()

    set_seed(args.seed)
    use_deepspeed = args.use_deepspeed
    # -------- Create one process and use later for broadcast later -------
    mag_tp_k_map = {}
    mag_tp_layer_map = {}

    # === Uncomment to do one gpu training ====
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_map = {
        'qwen': 'Qwen/Qwen2.5-1.5B',
        "qwen3":"Qwen/Qwen2.5-3B",
        "qwen7": "Qwen/Qwen2.5-7B",
        "qweni": "Qwen/Qwen2.5-1.5B-Instruct",
        'llama': 'meta-llama/Meta-Llama-3-8B',
        # 'gemma': 'google/gemma-3-1b-pt',
        'mistral': 'mistralai/Mistral-7B-v0.1',
        'smollm': 'HuggingFaceTB/SmolLM2-135M',
        'phi4': 'microsoft/phi-4',
        'llama3': 'meta-llama/Llama-3.2-3B',
        'smollm3':'HuggingFaceTB/SmolLM3-3B'
    }

    args.model_name = model_map[args.model]

    run_name = args.wandb_run_name
    if run_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"{args.model_name.split('/')[-1]}_{args.task}_{args.mode}_aggr{args.aggregation_mode}_lr{args.lr}_e{args.num_epoch}_s{args.seed}_{timestamp}"

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args)
        )

    if args.active_thresholds is not None:
        if len(args.active_thresholds) != 2:
            raise ValueError("Please provide exactly two values for --active_thresholds.")
        args.active_threshold = args.active_thresholds
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
            args.model_name
            # torch_dtype=torch.bfloat16
        )
    print(model.dtype)   
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id


    orig_param_count = sum(p.numel() for p in model.parameters())
    orig_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Loading dataset: {args.task}")
    if args.task == 'mquake':
        data_loader = MQUAKE(
            split_dir=dataset_paths['mquake'],
            chat_template=args.apply_chat_template,
            model_name=args.model_name
        )
    elif args.task == 'clutrr':
        data_loader = CLUTRR(
            split_dir=dataset_paths['clutrr'],
            chat_template=args.apply_chat_template,
            model_name=args.model_name
        )
    elif args.task == "tqa":
        data_loader = TQA(
            iti_split_dir=dataset_paths['tqa'],
            fold_num=args.tqa_fold_num,
            data_gen_seed=args.seed
        )
    elif args.task == 'musique':
        from diet.dataset.genloader import MuSiQue
        data_loader = MuSiQue(
            chat_template=args.apply_chat_template,
            model_name=args.model_name
        )
    elif args.task == 'babi':
        from diet.dataset.genloader import bAbI
        data_loader = bAbI(
            split_dir=dataset_paths['babi']
        )
    elif args.task == 'cogs':
        from diet.dataset.genloader import COGS
        data_loader = COGS(
            split_dir=dataset_paths['cogs']
        )
    elif args.task == 'gsm8k':
        from diet.dataset.genloader import GSM8K
        data_loader = GSM8K()
    elif args.task == 'boolq':
        data_loader = BoolQ(
            chat_template=args.apply_chat_template,
            model_name=args.model_name
        )
    elif args.task == 'rte':
        data_loader = RTE(
            chat_template=args.apply_chat_template,
            model_name=args.model_name
        )
    elif args.task == 'hellaswag':
        data_loader = HellaSwag(
            chat_template=args.apply_chat_template,
            model_name=args.model_name
        )
    elif args.task == 'winogrande':
        data_loader = WinoGrande(
            chat_template=args.apply_chat_template,
            model_name=args.model_name
        )
    elif args.task == 'arc-e':
        data_loader = ARC(
            subset="easy",
            chat_template=args.apply_chat_template,
            model_name=args.model_name
        )
    elif args.task == 'arc-c':
        data_loader = ARC(
            subset="challenge",
            chat_template=args.apply_chat_template,
            model_name=args.model_name
        )
    elif args.task == 'obqa':
        data_loader = OBQA(
            chat_template=args.apply_chat_template,
            model_name=args.model_name
        )
    elif args.task == 'logiqa2':
        data_loader = LogiQA2(
            chat_template=args.apply_chat_template,
            model_name=args.model_name
        )

    else:
        raise ValueError(f"Unsupported task: {args.task}")

    datasets = data_loader.load_data(train_size=args.train_size)

    if args.dev_mode:
        if args.task == 'musique':
            train_dataset = datasets['train']
            test_dataset = datasets['val']
        elif args.task == 'gsm8k':
            train_dataset = datasets['train']
            test_dataset = datasets['test']
        elif args.task in ['rte', 'hellaswag', 'winogrande']:
            train_dataset = datasets['train']
            test_dataset = datasets['test']  # these use validation as test
        elif args.task in ['arc-e', 'arc-c', 'obqa', 'logiqa2']:
            train_dataset = datasets['train']
            test_dataset = datasets['validation'] 
        else:
            dev_dataset, train_dataset = data_loader.get_dev_set(ratio=args.topk_ratio, return_rest=True)
            test_dataset = dev_dataset
    else:
        if args.task == 'musique':
            train_dataset = datasets['train']
            test_dataset = datasets['val']
        elif args.task == 'gsm8k':
            train_dataset = datasets['train']
            test_dataset = datasets['test']
        elif args.task in ['boolq', 'rte', 'hellaswag', 'winogrande']:
            train_dataset = datasets['train']
            test_dataset = datasets['test']  # these use validation as test
        elif args.task in ['arc-e', 'arc-c', 'obqa', 'logiqa2']:
            train_dataset = datasets['train']
            test_dataset = datasets['test']  # these have separate test splits
        else:
            train_dataset = datasets['train']
            test_dataset = datasets['test']  # default for mquake, clutrr, tqa, babi, cogs

    if args.task in ['mquake', 'musique', 'gsm8k', 'boolq', 'rte']:
        response_template_with_context = " A:\n"
    elif args.task == 'clutrr':
        response_template_with_context = " 's\n"
    elif args.task in ['hellaswag', 'winogrande', 'arc-e', 'arc-c', 'obqa', 'logiqa2']:
        response_template_with_context = "Answer:\n"
    elif args.task in ['babi', 'cogs']:
        response_template_with_context = " A:"
    elif args.task == 'tqa':
        response_template_with_context = " Answer:"
    else:
        response_template_with_context = "\n"  # Default fallback

    if args.apply_chat_template:
        response_template_with_context = \
            tokenizer.apply_chat_template([], tokenize=False, add_generation_prompt=True).split('\n')[-1]  # Heuristic
        if not response_template_with_context: response_template_with_context = " [/INST]"  # Fallback

    offset = 1  # Adjust based on tokenizer behavior if needed
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[offset:]
    if not use_deepspeed:
        model = model.to(device)
    optimizer = None
    pruner_stats = None
    if args.mode == "calibration":
        print("--- [MODE: ANALYSIS] Running Sub-network Stability Analysis ---")
        
        # 1. Define the calibration set ratios to test
        ratios_to_test = [0.01, 0.02, 0.05, 0.1]
        results_by_ratio = {} 

        print(f"[*] Testing active set stability for ratios: {ratios_to_test}")

        # 2. Loop through each ratio to find the active neurons
        for ratio in ratios_to_test:
            print(f"\n----- Processing ratio: {ratio} -----")
            # Get a small subset of data for calibration
            sampled_subset = data_loader.get_active_set(ratio)
            sampled_subset = sampled_subset['text']
            active_dataloader = torch.utils.data.DataLoader(sampled_subset, batch_size=args.eval_batch)
            print(f"[MEM] Before processing ratio {ratio}: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")               
            # Initialize the tracker to find active neurons
            tracker = NeuronTracker(
                model=model,
                tokenizer=tokenizer,
                threshold=args.active_threshold,
                topk_ratio=args.topk_ratio, 
                use_abs_threshold=args.use_abs_threshold,
                device=device,
                track_attention_proj=args.tune_attn,
                verbose=False # Keep the log clean during the loop
            )
            
            # Get and store the dictionary of active neurons for this ratio
            active_indices_dict = tracker.get_active_indices(dataloader=active_dataloader)
            if active_indices_dict:
                num_layers = len(active_indices_dict) - 1 if '_attn_proj_layers' in active_indices_dict else len(active_indices_dict)
                print(f"[*] Found active indices for {num_layers} layers.")
                results_by_ratio[ratio] = active_indices_dict
                print(f"[MEM] After storing ratio {ratio}: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")
            else:
                print("[!] Warning: No active neurons found for this ratio.")
        
        # 3. Calculate the pairwise similarity matrix
        print("\n[*] Calculating similarity matrix for the heatmap...")
        tested_ratios = sorted(results_by_ratio.keys())
        num_ratios = len(tested_ratios)
        similarity_matrix = np.zeros((num_ratios, num_ratios))

        for i in range(num_ratios):
            for j in range(num_ratios):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                    continue
                
                r1 = tested_ratios[i]
                r2 = tested_ratios[j]
                dict1 = results_by_ratio[r1]
                dict2 = results_by_ratio[r2]
                
                similarity = calculate_jaccard_similarity(dict1, dict2)
                similarity_matrix[i, j] = similarity
        
        # 4. Generate and save the heatmap visualization
        print("[*] Generating and saving heatmap...")

        N = similarity_matrix.shape[0]

        # 画布尺寸随矩阵大小自适应：格子太少就放大，格子多就控制不要太挤
        fig_w = max(1.6 * N, 6.0)
        fig_h = fig_w
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        # 字号自适应（N 小 → 字大；N 大 → 字缩小，避免重叠）
        tick_fs = max(8, int(14 - 0.3 * N))
        ann_fs  = max(8, int(12 - 0.3 * N))

        hm = sns.heatmap(
            similarity_matrix,
            annot=True, fmt=".3f",
            annot_kws={"fontsize": ann_fs, "fontweight": "bold"},  # 粗体更清晰
            xticklabels=tested_ratios,
            yticklabels=tested_ratios,
            cmap="viridis",
            vmin=0.5, vmax=1.0,
            square=True,
            linewidths=1.0, linecolor="white",
            cbar=True,
            cbar_kws=dict(fraction=0.035, pad=0.02, shrink=0.9, aspect=40, ticks=[0.5, 0.75, 1.0]),
            ax=ax
        )

        # 放大轴刻度和色条刻度
        cbar = hm.collections[0].colorbar
        cbar.ax.tick_params(labelsize=tick_fs)
        ax.tick_params(axis="x", labelrotation=0, labelsize=tick_fs, length=6, width=1.2)
        ax.tick_params(axis="y", labelsize=tick_fs, length=6, width=1.2)

        ax.set_xlabel("Calibration Set Sample Ratio", fontsize=tick_fs + 2)
        ax.set_ylabel("Calibration Set Sample Ratio", fontsize=tick_fs + 2)

        plt.tight_layout()

        base = f"stability_heatmap_{str(args.model).replace('/', '_')}_{args.task}"
        plt.savefig(f"{base}.pdf", bbox_inches="tight")
        plt.savefig(f"{base}.svg", bbox_inches="tight")
        print(f"[*] Heatmap saved to: {base}.pdf / {base}.svg")
    if args.mode == "ns" or args.mode == "nslora":
        print("--- Starting Neuron Activation Tracking ---")
        print("[FLOPs] Measuring activation tracking cost...")
        # 1. Data Preparation
        sampled_subset = data_loader.get_active_set(args.active_sample_ratio)
        sampled_subset = sampled_subset['text']
        active_dataloader = torch.utils.data.DataLoader(sampled_subset, batch_size=args.eval_batch)
        # _ = peek_dataloader(active_dataloader, tokenizer=tokenizer, n_batches=2, prefix="[ACTIVE]")

        sel_stats = estimate_flops_infer(
            model=model,                
            data=active_dataloader,     
            modality="llm",
            tokenizer=tokenizer,              
            exclude_embeddings=True
        )
        print(f"[FLOPs] Selection (active set) ≈ {sel_stats['flops']/1e12:.3f} TFLOPs "
              f"[FLOPs] Selection (active set) ≈ {sel_stats['flops']/1e15:.3f} PFLOPs "
              f"(N={sel_stats['N']:,}, L=2048)")
        if args.use_wandb:
            wandb.log({
                "flops/selection_total": sel_stats["flops"],
                "flops/selection_total_tflops": sel_stats["flops"]/1e12,
                "flops/selection_N": sel_stats["N"],
            })

        # 2. NeuronTracker Initialization
        print("Initializing NeuronTracker...")
        tracker = NeuronTracker(
            model=model,  # Assuming 'model' is the loaded original model
            tokenizer=tokenizer,
            threshold=args.active_threshold,
            topk_ratio=args.topk_ratio, 
            use_abs_threshold=args.use_abs_threshold,
            device=device,
            track_attention_proj=args.tune_attn,
            verbose=True
        )

        # 3. Tracker Usage
        print("Running activation tracking...")
        layer_map = tracker.get_layer_name_map()
        active_indices_dict = None
        active_indices_dict = tracker.get_active_indices(dataloader=active_dataloader)
        if active_indices_dict is None:
            active_indices_dict = {}
        print(f"Tracking complete. Found active indices for "
                f"{len(active_indices_dict) - 1 if active_indices_dict and '_attn_proj_layers' in active_indices_dict else (len(active_indices_dict) if active_indices_dict else 0)} layers.")

        if len(active_indices_dict) == 0:
            print("No active neurons found — skipping pruning.")
            model_to_prune = model
        else:
            print("Loading fresh model instance for pruning...")
            model_to_prune = model


        # 4. NeuroselectiveTransformer5 Initialization
        print("Initializing NeuroselectiveTransformer5...")
        transformer = NeuroselectiveTransformer5(
            model=model_to_prune,
            active_neurons=active_indices_dict,
            layer_name_map=layer_map,
            verbose=True,
            tune_pruned=False,
            device=device
        )
        print("Performing model transformation (pruning)...")
        pruned_model = transformer.transform()
        pruned_model = pruned_model.to(device)
        model = pruned_model
        print("Model variable now points to the pruned model.")

        # For nslora mode, apply NSLoraLinear to NeuroselectiveLinear layers
        if args.mode == "nslora":
            print("--- Applying NSLoraLinear to NeuroselectiveLinear layers ---")
            # Convert NeuroselectiveLinear layers to NSLoraLinear
            for name, module in model.named_modules():
                if isinstance(module, NeuroselectiveLinear):
                    parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                    attr_name = name.split('.')[-1]

                    # Get parent module
                    parent = model
                    if parent_name:
                        for part in parent_name.split('.'):
                            parent = getattr(parent, part)

                    # Replace NeuroselectiveLinear with NSLoraLinear
                    ns_lora = NSLoraLinear(
                        ns_linear=module,
                        r=args.lora_r,
                        lora_alpha=args.lora_alpha,
                        lora_dropout=args.lora_dropout,
                        merge_weights=False
                    )

                    # Set the attribute on the parent module
                    setattr(parent, attr_name, ns_lora)

            print("NSLoraLinear transformation complete.")
            _bp = next(p for p in model.parameters() if p.is_floating_point())  # [ADDED]
            for m in model.modules():  # [ADDED]
                if isinstance(m, NSLoraLinear):  # [ADDED]
                    m.lora_A.to(device=_bp.device, dtype=_bp.dtype)  # [ADDED]
                    m.lora_B.to(device=_bp.device, dtype=_bp.dtype)  # [ADDED]
                    m.scaling = torch.as_tensor(m.scaling, device=_bp.device, dtype=_bp.dtype)  # [ADDED]
            print("[NSLoRA] base:", _bp.dtype, "adapters:", {m.lora_A.weight.dtype for m in model.modules() if isinstance(m, NSLoraLinear)})  # [ADDED]
    elif args.mode == "covplot":
        # ===== NS layerwise cross-task analysis for LLM (4x4, hardcoded rho) =====
        from collections import OrderedDict
        from matplotlib import colors, cm

        # ---- 任务与 ρ（请按你的设定填写具体数值）----
        TASK_SPECS = OrderedDict({
            "arc-e":   {"rho": 0.9},   # ← 填你的 rho
            "arc-c":   {"rho": 0.9},   # ← 填你的 rho
            "boolq":   {"rho": 0.8},   # ← 填你的 rho
            "clutrr":  {"rho": 0.01},   # ← 填你的 rho
        })

        ratio = 0.01
        print(f"[covplot] tasks: " + ", ".join(f"{k}:{v['rho']}" for k,v in TASK_SPECS.items())
            + f" | calibration sample ratio={ratio}")

        # ---- 每个任务各做一次选择（non-shuffle loader → active 子集 → NeuronTracker）----
        results_by_task = OrderedDict()
        for ds_name, spec in TASK_SPECS.items():
            print(f"\n----- [covplot] Processing: {ds_name} (rho={spec['rho']}) -----")
            
            # 创建对应的 data_loader
            if ds_name == 'clutrr':
                temp_data_loader = CLUTRR(
                    split_dir=dataset_paths['clutrr'],
                    chat_template=args.apply_chat_template,
                    model_name=args.model_name
                )
            elif ds_name == 'arc-e':
                temp_data_loader = ARC(
                    subset="easy",
                    chat_template=args.apply_chat_template,
                    model_name=args.model_name
                )
            elif ds_name == 'arc-c':
                temp_data_loader = ARC(
                    subset="challenge",
                    chat_template=args.apply_chat_template,
                    model_name=args.model_name
                )
            elif ds_name == 'boolq':
                temp_data_loader = BoolQ(
                    chat_template=args.apply_chat_template,
                    model_name=args.model_name
                )
            else:
                raise ValueError(f"Unsupported dataset: {ds_name}")
                
            sampled_subset = temp_data_loader.get_active_set(ratio)
            sampled_subset = sampled_subset['text']
            active_dataloader = torch.utils.data.DataLoader(sampled_subset, batch_size=args.eval_batch)

            thr = args.threshold if hasattr(args, "threshold") else getattr(args, "active_threshold", 0.01)
            tracker = NeuronTracker(
                model=model, tokenizer=tokenizer, threshold=thr,
                topk_ratio=float(spec["rho"]), use_abs_threshold=args.use_abs_threshold,
                device=device, track_attention_proj=args.tune_attn, verbose=False
            )
            active_neurons = tracker.get_active_indices(dataloader=active_dataloader)
            if active_neurons:
                print(f"[*] [{ds_name}] active layers: {len(active_neurons)}")
                results_by_task[ds_name] = active_neurons
            else:
                print(f"[!] [{ds_name}] empty selection; recording empty dict")
                results_by_task[ds_name] = {}

        task_names = list(results_by_task.keys())
        N = len(task_names)
        if N < 2:
            raise RuntimeError("[covplot] need at least 2 tasks")

        # ---- 计算 Cov(A|B)（行=Target A, 列=Source B）与 Jaccard ----
        print("\n[*] [covplot] Computing matrices ...")
        cov_matrix  = np.zeros((N, N), dtype=float)
        jacc_matrix = np.zeros((N, N), dtype=float)
        for i, A in enumerate(task_names):
            for j, B in enumerate(task_names):
                cov_matrix[i, j]  = calculate_directed_coverage(results_by_task[A], results_by_task[B], weighted=True)
                jacc_matrix[i, j] = calculate_jaccard_similarity(results_by_task[A], results_by_task[B])

        # ---- 画图（极简；无自定义函数；两次重复画法）----
        model_tag = args.model_name.replace("/", "_")
        base_tag  = f"llm_cross_task_{model_tag}"
        tick_labels = [t.replace("_", "\n").upper() for t in task_names]

        # 公共绘图参数
        fig_w = max(1.2 * N, 3.8)
        fig_h = max(1.2 * N, 3.8)
        tick_fs = max(8, int(14 - 0.3 * N))
        ann_fs  = max(8, int(12 - 0.3 * N))
        vmin, vmax = 0.0, 1.0

        # 1) Cov(A|B)
        plt.figure(figsize=(fig_w, fig_h))
        ax = sns.heatmap(
            cov_matrix, annot=False, xticklabels=tick_labels, yticklabels=tick_labels,
            cmap="viridis", vmin=vmin, vmax=vmax, linewidths=1.0, linecolor="white",
            square=True, cbar=True, cbar_kws=dict(fraction=0.03, pad=0.02, shrink=0.9, aspect=40, ticks=[0.0, 0.5, 1.0])
        )
        ax.tick_params(axis="x", labelrotation=0, pad=6, length=6, width=1.1, direction="out", labelsize=tick_fs)
        ax.tick_params(axis="y", pad=6, length=6, width=1.1, direction="out", labelsize=tick_fs)
        ax.set_xlabel("SOURCE TASK", fontsize=max(10, tick_fs))
        ax.set_ylabel("TARGET TASK", fontsize=max(10, tick_fs))
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-.5, N, 1), minor=True)
        ax.set_yticks(np.arange(-.5, N, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)
        # 居中数字 + 亮度自适应黑/白
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        _cmap = cm.get_cmap("viridis")
        for i in range(N):
            for j in range(N):
                val = cov_matrix[i, j]
                rgba = _cmap(norm(val))
                lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                txt_color = "white" if lum < 0.5 else "black"
                ax.text(j + 0.5, i + 0.5, f"{val:.3f}", ha="center", va="center", fontsize=ann_fs, color=txt_color, clip_on=True, zorder=3)
        ax.set_xlim(0, N); ax.set_ylim(N, 0)
        plt.tight_layout()
        plt.savefig(f"cov_heatmap_{base_tag}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"cov_heatmap_{base_tag}.pdf", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[*] Saved: cov_heatmap_{base_tag}.png/.pdf")

        # 2) Jaccard
        plt.figure(figsize=(fig_w, fig_h))
        ax = sns.heatmap(
            jacc_matrix, annot=False, xticklabels=tick_labels, yticklabels=tick_labels,
            cmap="viridis", vmin=vmin, vmax=vmax, linewidths=1.0, linecolor="white",
            square=True, cbar=True, cbar_kws=dict(fraction=0.03, pad=0.02, shrink=0.9, aspect=40, ticks=[0.0, 0.5, 1.0])
        )
        ax.tick_params(axis="x", labelrotation=0, pad=6, length=6, width=1.1, direction="out", labelsize=tick_fs)
        ax.tick_params(axis="y", pad=6, length=6, width=1.1, direction="out", labelsize=tick_fs)
        ax.set_xlabel("SOURCE TASK", fontsize=max(10, tick_fs))
        ax.set_ylabel("TARGET TASK", fontsize=max(10, tick_fs))
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-.5, N, 1), minor=True)
        ax.set_yticks(np.arange(-.5, N, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        _cmap = cm.get_cmap("viridis")
        for i in range(N):
            for j in range(N):
                val = jacc_matrix[i, j]
                rgba = _cmap(norm(val))
                lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                txt_color = "white" if lum < 0.5 else "black"
                ax.text(j + 0.5, i + 0.5, f"{val:.3f}", ha="center", va="center", fontsize=ann_fs, color=txt_color, clip_on=True, zorder=3)
        ax.set_xlim(0, N); ax.set_ylim(N, 0)
        plt.tight_layout()
        plt.savefig(f"jacc_heatmap_{base_tag}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"jacc_heatmap_{base_tag}.pdf", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[*] Saved: jacc_heatmap_{base_tag}.png/.pdf")

        print("\n[INFO] LLM covplot (4x4) complete.")
        return
    
    elif args.mode == "transfer":
        src_name = args.source_task
        src_topk = args.source_ratio

        # print(f"[transfer] source={src_name}  ρ={src_topk:.2f}  sample_ratio={sel_ratio}")

        # 创建源任务的 data_loader
        if src_name == 'clutrr':
            data_loader = CLUTRR(
                split_dir=dataset_paths['clutrr'],
                chat_template=args.apply_chat_template,
                model_name=args.model_name
            )
        elif src_name == 'arc-e':
            data_loader = ARC(
                subset="easy",
                chat_template=args.apply_chat_template,
                model_name=args.model_name
            )
        elif src_name == 'arc-c':
            data_loader = ARC(
                subset="challenge",
                chat_template=args.apply_chat_template,
                model_name=args.model_name
            )
        elif src_name == 'boolq':
            data_loader = BoolQ(
                chat_template=args.apply_chat_template,
                model_name=args.model_name
            )
        else:
            raise ValueError(f"Unsupported source task: {src_name}")

        sampled_subset = data_loader.get_active_set(args.active_sample_ratio)
        sampled_subset = sampled_subset['text']
        src_active_dl = torch.utils.data.DataLoader(sampled_subset, batch_size=args.eval_batch)
        tracker_src = NeuronTracker(
            model=model,
            tokenizer=tokenizer,                
            threshold=args.active_threshold,
            topk_ratio=args.source_ratio,                
            use_abs_threshold=args.use_abs_threshold,
            device=device,
            track_attention_proj=args.tune_attn,
            verbose=True
        )
        active_src   = tracker_src.get_active_indices(dataloader=src_active_dl) or {}
        layer_map_src = tracker_src.get_layer_name_map()

        nst = NeuroselectiveTransformer5(
            model=model,
            active_neurons=active_src,
            layer_name_map=layer_map_src,
            verbose=True,
            tune_pruned=False,
            device=device
        )
        model = nst.transform().to(device)
        print("[transfer] model pruned by SOURCE-selected subnetwork; downstream stays unchanged.")

        # 4) （可选）记录统计/开销（不影响训练逻辑）
        try:
            stats = nst.get_parameter_stats()
            print(f"[ns] overall_reduction = {stats['overall_model_reduction_perc']:.2f}%")
            sel_samples = len(src_active_dl.dataset)
            F_fwd_base  = flops_forward(model, _inp1_fwd, device=str(device))
            print(f"[FLOPs] source-selection forward ({sel_samples} ex): {sel_samples*F_fwd_base/1e9:.2f} GFLOPs")
        except Exception:
            pass
    
    elif args.mode == "mag_pt":
        sampled_subset = data_loader.get_active_set(args.active_sample_ratio)
        sampled_subset = sampled_subset['text']
        active_dataloader = torch.utils.data.DataLoader(sampled_subset, batch_size=args.eval_batch)
        # _ = peek_dataloader(active_dataloader, tokenizer=tokenizer, n_batches=2, prefix="[ACTIVE]")

        sel_stats = estimate_flops_infer(
            model=model,                
            data=active_dataloader,     
            modality="llm",
            tokenizer=tokenizer,              
            exclude_embeddings=True
        )
        # 2. NeuronTracker Initialization
        print("Initializing NeuronTracker...")
        tracker = NeuronTracker(
            model=model,  # Assuming 'model' is the loaded original model
            tokenizer=tokenizer,
            threshold=args.active_threshold,
            topk_ratio=args.topk_ratio, 
            use_abs_threshold=args.use_abs_threshold,
            device=device,
            track_attention_proj=args.tune_attn,
            verbose=True
        )

        # 3. Tracker Usage
        print("Running activation tracking...")
        layer_map = tracker.get_layer_name_map()
        active_indices_dict = None
        active_indices_dict = tracker.get_active_indices(dataloader=active_dataloader)
        if active_indices_dict is None:
            active_indices_dict = {}
        print(f"Tracking complete. Found active indices for "
              f"{len(active_indices_dict) - 1 if active_indices_dict and '_attn_proj_layers' in active_indices_dict else (len(active_indices_dict) if active_indices_dict else 0)} layers.")

        k_map = {}
        for ln, idx in active_indices_dict.items():
            try:
                k_map[ln] = int(idx.numel())
            except Exception:
                k_map[ln] = int(len(idx))

        mag_indices = {}
        for m in model.modules():
            if isinstance(m, nn.Linear):
                lname = layer_map.get(m, None)
                if lname is None:
                    continue
                if not any(tag in lname for tag in ["gate_proj", "wi", "lin1"]):
                    continue
                k = k_map.get(lname, 0)
                if k <= 0:
                    continue
                # weight: [out_features, in_features] → L1 per output unit
                score = m.weight.detach().abs().sum(dim=1)
                keep = min(k, score.numel())
                topk_idx = torch.topk(score, keep, largest=True).indices
                mag_indices[lname] = topk_idx

        # 3) structural pruning into a neuroselective subnetwork
        nst = NeuroselectiveTransformer5(
            model=model,
            active_neurons=mag_indices,
            layer_name_map=layer_map,
            tune_pruned=False,
            device=device,
            verbose=True
        )
        model = nst.transform().to(device)

        # 4) freeze all, then unfreeze pruned-MLP (NeuroselectiveLinear) + output head
        for p in model.parameters():
            p.requires_grad = False

        # heads typically named "lm_head" (plus a few common fallbacks)
        trainable_params_list = []
        for name, p in model.named_parameters():
            p.requires_grad = False
        for name, module in model.named_modules():
            if isinstance(module, NeuroselectiveLinear) or ('lm_head' in name):
                for p in module.parameters():
                    p.requires_grad = True
                    trainable_params_list.append(p)
        if not trainable_params_list:
            raise RuntimeError("[mag_pt] no NeuroselectiveLinear/lm_head found as trainables.")
        optimizer = AdamW(trainable_params_list, lr=args.lr, weight_decay=args.wd)

        # 5) accounting + fresh optimizer (downstream code expects `optimizer`)
        final_param_count = sum(p.numel() for p in model.parameters())
        final_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[mag_pt] Final model parameters: {final_param_count:,}")
        print(f"[mag_pt] Final trainable parameters: {final_trainable_params:,}")
    elif args.mode == "wanda_p":
        sampled_subset = data_loader.get_active_set(args.active_sample_ratio)
        sampled_subset = sampled_subset['text']
        active_dataloader = torch.utils.data.DataLoader(sampled_subset, batch_size=args.eval_batch)

        sel_stats = estimate_flops_infer(
            model=model,                
            data=active_dataloader,     
            modality="llm",
            tokenizer=tokenizer,              
            exclude_embeddings=True
        )

        print("Initializing NeuronTracker (budget)...")
        tracker_budget = NeuronTracker(
            model=model,
            tokenizer=tokenizer,
            threshold=args.active_threshold,
            topk_ratio=args.topk_ratio,           
            use_abs_threshold=args.use_abs_threshold,
            device=device,
            track_attention_proj=args.tune_attn,
            verbose=True
        )

        print("Running activation tracking for budget...")
        layer_map = tracker_budget.get_layer_name_map()
        active_indices_dict = tracker_budget.get_active_indices(dataloader=active_dataloader) or {}
        print(f"[wanda_p] active layers (raw): {len(active_indices_dict)}")

        k_map = {}
        for ln, idx in active_indices_dict.items():
            try:
                k_map[ln] = int(idx.numel())
            except Exception:
                k_map[ln] = int(len(idx))

        print("Running Wanda ranking (L1  activation)...")
        tracker_w = NeuronTracker(
            model=model,
            tokenizer=tokenizer,
            topk_ratio=1.0,
            device=device,
            track_attention_proj=args.tune_attn,
            verbose=True
        )
        wanda_calib_batches = int(getattr(args, "wanda_calib_batches", 1))
        wanda_indices_all = tracker_w.get_wanda_indices(
            dataloader=active_dataloader,
            scan_batches=wanda_calib_batches
        )

        wanda_indices = {}
        for m in model.modules():
            if not isinstance(m, nn.Linear):
                continue
            lname = layer_map.get(m, None)
            if lname is None:
                continue
            if not any(tag in lname for tag in ["gate_proj", "wi", "lin1"]):
                continue  

            idx = wanda_indices_all.get(lname, None)
            if idx is None or len(idx) == 0:
                continue

            k = int(k_map.get(lname, 0))
            if k <= 0:
                continue

            keep = min(k, len(idx)) 
            topk_idx = torch.as_tensor(idx[:keep], dtype=torch.long)
            wanda_indices[lname] = topk_idx

        for ln, ids in wanda_indices.items():
            assert len(ids) == min(k_map.get(ln, 0), len(wanda_indices_all.get(ln, []))), \
                f"[wanda_p] budget mismatch @ {ln}: want={k_map.get(ln,0)} got={len(ids)}"

        nst = NeuroselectiveTransformer5(
            model=model,
            active_neurons=wanda_indices,
            layer_name_map=layer_map,
            tune_pruned=False,
            device=device,
            verbose=True
        )
        model = nst.transform().to(device)

        for p in model.parameters():
            p.requires_grad = False

        trainable_params_list = []
        for name, p in model.named_parameters():
            p.requires_grad = False
        for name, module in model.named_modules():
            if isinstance(module, NeuroselectiveLinear) or ('lm_head' in name):
                for p in module.parameters():
                    p.requires_grad = True
                    trainable_params_list.append(p)

        if not trainable_params_list:
            raise RuntimeError("[wanda_p] no NeuroselectiveLinear/lm_head found as trainables.")

        optimizer = AdamW(trainable_params_list, lr=args.lr, weight_decay=args.wd)

        final_param_count = sum(p.numel() for p in model.parameters())
        final_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[wanda_p] Final model parameters: {final_param_count:,}")
        print(f"[wanda_p] Final trainable parameters: {final_trainable_params:,}")
    elif args.mode == "wanda_tp":
        sampled_subset = data_loader.get_active_set(args.active_sample_ratio)['text']
        active_dataloader = torch.utils.data.DataLoader(sampled_subset, batch_size=args.eval_batch, shuffle=False)
        tracker = NeuronTracker(model=model, tokenizer=tokenizer,
                             threshold=args.active_threshold, topk_ratio=args.topk_ratio,
                             use_abs_threshold=args.use_abs_threshold, device=device,
                             track_attention_proj=args.tune_attn, verbose=True)
        layer_map = tracker.get_layer_name_map()
        active_indices = tracker.get_active_indices(dataloader=active_dataloader) or {}
        k_map = {ln: int(getattr(idx, "numel", lambda: len(idx))()) for ln, idx in active_indices.items()}
        mag_tp_layer_map = layer_map
        mag_tp_k_map = k_map
        optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.wd)

    elif args.mode == "mag_tp":
        sampled_subset = data_loader.get_active_set(args.active_sample_ratio)
        sampled_subset = sampled_subset['text']
        active_dataloader = torch.utils.data.DataLoader(
             sampled_subset, batch_size=args.eval_batch, shuffle=False
         )
         # （可选）记录 selection FLOPs，接口与 mag_pt 一致
        _ = estimate_flops_infer(
             model=model, data=active_dataloader, modality="llm",
             tokenizer=tokenizer, exclude_embeddings=True
        )
         # 跟 mag_pt 相同的 tracker 配置，但这里只做 k-map
        tracker = NeuronTracker(
             model=model, tokenizer=tokenizer,
             threshold=args.active_threshold, topk_ratio=args.topk_ratio,
             use_abs_threshold=args.use_abs_threshold, device=device,
             track_attention_proj=args.tune_attn, verbose=True
        )
        layer_map = tracker.get_layer_name_map()
        active_indices = tracker.get_active_indices(dataloader=active_dataloader) or {}
        k_map = {}
        for ln, idx in active_indices.items():
            try:
                k_map[ln] = int(idx.numel())
            except Exception:
               k_map[ln] = int(len(idx))
        mag_tp_k_map = k_map
        mag_tp_layer_map = layer_map
        print(f"[mag_tp][pre] captured budget for {len(k_map)} layers, e.g. {list(k_map.items())[:3]}")
        optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                           lr=args.lr, weight_decay=args.wd)

    elif args.mode =="lora":
        print(f"--- Setting up {args.mode} ---")
        lora_config_dict = {
            'type': args.mode.lower(),           
            'r': args.lora_r,
            'alpha': args.lora_alpha,            
            'dropout': args.lora_dropout,        
            'bias': 'none',
            'task_type': 'CAUSAL_LM',
        }
        model = setup_lora(model, lora_config_dict)  # Assumes setup_lora handles different PEFT types
        if not use_deepspeed:
            model = model.to(device)
        from peft.tuners.lora import LoraLayer
        base_dt = next(p for p in model.parameters() if p.is_floating_point()).dtype
        print("[PEFT] base:", base_dt, "adapters:", {next(iter(m.lora_A.values())).weight.dtype for m in model.modules() if isinstance(m, LoraLayer)})

    else:
        print(f"--- Running in {args.mode} mode (full finetuning) ---")
        if not use_deepspeed:
            model = model.to(device)
        print(model.dtype)   

    final_param_count = sum(p.numel() for p in model.parameters())
    final_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.mode in ("ns", "transfer"):
        trainable_params_list = []
        for name, param in model.named_parameters():
            param.requires_grad = False
        for name, module in model.named_modules():
            if isinstance(module, NeuroselectiveLinear) or 'lm_head' in name:
                for param in module.parameters():
                    param.requires_grad = True
                    trainable_params_list.append(param)
        if not use_deepspeed:
            optimizer = AdamW(trainable_params_list, lr=args.lr, weight_decay=args.wd)
    elif args.mode == "nslora":
        if isinstance(model.lm_head, torch.nn.Linear):
            head_cfg = LoraConfig(
                r=args.lora_r,          
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=["lm_head"],   
                bias="none",
            )
            model = get_peft_model(model, head_cfg)        

        # For NSLoraLinear, we only train the LoRA parameters
        trainable_params_list = []
        for name, param in model.named_parameters():
            param.requires_grad = False

        # Enable training for NSLoraLinear parameters and lm_head
        for name, module in model.named_modules():
            if isinstance(module, NSLoraLinear) or 'lm_head' in name:
                for param_name, param in module.named_parameters():
                    if 'lora_A' in param_name or 'lora_B' in param_name:
                        param.requires_grad = True
                        trainable_params_list.append(param)
        if not trainable_params_list:
            raise RuntimeError("no lora is found, check")

        # ① print trainable
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in trainable_params_list)
        print(f"[NS-LoRA] trainable: {trainable_params:,} / {total_params:,}  "
            f"({trainable_params / total_params * 100:.3f}%)")

        # ② print shape
        print("lora list:")
        for name, param in model.named_parameters():
            if param.requires_grad:          # 只会列出 lora_A / lora_B
                print(f"  {name:60s}  {tuple(param.shape)}")
        if not use_deepspeed:
            optimizer = AdamW(trainable_params_list, lr=args.lr, weight_decay=args.wd)
    else:
        if not use_deepspeed:
            optimizer = AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.lr,
                weight_decay=args.wd
            )

    print("\n--- Parameter Counts ---")
    print(f"Original Total Params:      {orig_param_count:,}")
    print(f"Original Trainable Params:  {orig_trainable_params:,}")
    final_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    final_total_params     = sum(p.numel() for p in model.parameters())
    print(f"Final Total Params:         {final_total_params:,}")
    print(f"Final Trainable Params:     {final_trainable_params:,}")
    if orig_trainable_params > 0:
        reduction = (1 - final_trainable_params / orig_trainable_params) * 100
        print(f"Trainable Param Reduction:  {reduction:.2f}%")
    else:
        print("No original trainable parameters to compare.")
    print("-" * 26)

    if args.use_wandb:
        log_data = {
            'params/original_total': orig_param_count,
            'params/original_trainable': orig_trainable_params,
            'params/final_total': final_param_count,
            'params/final_trainable': final_trainable_params,
        }
        if orig_trainable_params > 0:
            log_data['params/trainable_reduction_pct'] = (1 - final_trainable_params / orig_trainable_params) * 100
        if pruner_stats:
            log_data.update({f'pruner/{k}': v for k, v in pruner_stats.items() if not isinstance(v, (dict, list))})
            if 'overall_model_reduction_perc' in pruner_stats:
                log_data['params/overall_reduction_pct'] = pruner_stats['overall_model_reduction_perc']

        wandb.log(log_data)

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template_ids,
        tokenizer=tokenizer
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epoch,
        per_device_train_batch_size=args.train_batch,
        per_device_eval_batch_size=args.eval_batch,
        seed=args.seed,
        eval_strategy="epoch",
        save_strategy="no",
        lr_scheduler_type="linear" if args.schedule else "constant",
        load_best_model_at_end=False,
        max_length=256,
        dataset_text_field="text",
        bf16=False,
        warmup_ratio=0.1,      
        learning_rate=args.lr,
        weight_decay=args.wd,                 
        max_grad_norm=0.5,                               
        deepspeed=(args.deepspeed if use_deepspeed else None),
    )
    if args.task == "clutrr":
        eval_dataset = datasets['val']
    else:
        eval_dataset = test_dataset

    trainer = CustomSFTTrainerV2(
        task=args.task,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, None) if not use_deepspeed else (None, None)
    )

    if args.use_wandb:
        wandb.log({"status": "training_started"})
    
    try:
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.train_batch, shuffle=False
        )
        tr_stats = estimate_flops_train(
            model=model, data=train_dataloader, modality="llm",
            epochs=args.num_epoch, tokenizer=tokenizer
        )
        print(f"[FLOPs] Train (planned) ≈ {tr_stats['flops']/1e15:.3f} PFLOPs "
              f"[FLOPs] Train (planned) ≈ {tr_stats['flops']/1e12:.3f} TFLOPs "
              f"(N={tr_stats['N']:,}, MAX_TOKEN=2048, epochs={args.num_epoch})")
        if args.use_wandb:
            wandb.log({
                "flops/train_total": tr_stats["flops"],
                "flops/train_total_tflops": tr_stats["flops"]/1e12,
                "flops/train_total_pflops": tr_stats["flops"]/1e15,
            })
    except Exception as e:
        logging.warning(f"[FLOPs] Train estimation failed: {e}")

    print("Starting training...")
    trainer.train()


    if args.use_wandb:
        wandb.log({"status": "training_finished", "evaluating": True})

    accuracy, predictions = trainer.test(
        fname=os.path.join(args.output_dir, run_name),
        task=args.task,
        eval_dataset=test_dataset,
        model_name=args.model_name,
        apply_chat_template=args.apply_chat_template,
    )
    eval_dataloader = DataLoader(
            test_dataset, batch_size=args.eval_batch, shuffle=False
        )

    inf_unmerged = estimate_flops_infer(
        model=model,
        data=eval_dataloader,   
        modality="llm",
        tokenizer=tokenizer,   
        exclude_embeddings=True
    )

    # ---- Eval FLOPs ----
    total_eval_flops = inf_unmerged["flops"]
    N_eval = max(inf_unmerged["N"], 1)
    assert inf_unmerged["D"] == 2048 * inf_unmerged["N"], \
        "Eval D != N*2048 — check dataloader/iterator & tokenizer."
    print(f"[FLOPs] Inference (eval, unmerged) ≈ {total_eval_flops/1e15:.3f} PFLOPs "
          f"({total_eval_flops/1e12:.3f} TFLOPs); "
          f"per-example ≈ {(total_eval_flops/N_eval)/1e9:.3f} GFLOPs "
          f"(L=2048, N={inf_unmerged['N']:,})")
    if args.use_wandb:
        wandb.log({
            "flops/eval_total": total_eval_flops,
            "flops/eval_total_pflops": total_eval_flops/1e15,
            "flops/eval_total_tflops": total_eval_flops/1e12,
            "flops/eval_per_example_gflops": (total_eval_flops/N_eval)/1e9,
            "tokens/eval_D": inf_unmerged["D"],
            "tokens/eval_L": 2048,
            "size/eval_N": inf_unmerged["N"],
        })





    if args.use_wandb:
        wandb.log({
            "flops/infer_eval_unmerged_2ND": inf_unmerged["flops"],
            "tokens/eval_D": inf_unmerged["D"],
            "params/nonembed_postprune": inf_unmerged["N"],
        })

    print("\n[Evaluation Metrics]")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Num predictions: {len(predictions)}")
    print("\n[Evaluation Metrics]")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Num predictions: {len(predictions)}")
    
    if args.use_wandb:
        wandb.log({
            "eval/accuracy": accuracy,
            "eval/num_predictions": len(predictions)
        })

    if args.mode in ("mag_tp", "wanda_tp"):
        k_map = mag_tp_k_map or {}
        layer_map = mag_tp_layer_map or {}
        if not k_map or not layer_map:
            raise RuntimeError("[mag_tp] Budget map not found; ensure pre-phase ran.")
        prune_indices = {}
        if args.mode == "wanda_tp":
          sampled_subset = data_loader.get_active_set(args.active_sample_ratio)['text']
          active_dataloader = torch.utils.data.DataLoader(sampled_subset, batch_size=args.eval_batch, shuffle=False)
          tracker_w = NeuronTracker(model=model, tokenizer=tokenizer, topk_ratio=1.0,
                                   device=device, track_attention_proj=args.tune_attn, verbose=True)
          wanda_all = tracker_w.get_wanda_indices(
              dataloader=active_dataloader,
              scan_batches=int(getattr(args, "wanda_calib_batches", 1))
          )
          sel_indices = {}
          for m in model.modules():
              if isinstance(m, nn.Linear):
                  lname = layer_map.get(m, None)
                  if lname and any(t in lname for t in ["gate_proj","up_proj","wi","fc1","lin1"]):
                      k = int(k_map.get(lname, 0))
                      idx = wanda_all.get(lname, [])[:max(0, k)]
                      if k > 0 and len(idx) > 0:
                          prune_indices[lname] = torch.as_tensor(idx, dtype=torch.long)

        else:
            print(f"[mag_tp] Using budget for {len(k_map)} layers")
            print("[mag_tp] Computing magnitude-based pruning indices...")
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    lname = layer_map.get(m, None)
                    if lname is None or not any(tag in lname for tag in ["gate_proj","up_proj","wi","fc1","lin1"]):
                        continue
                    k = int(k_map.get(lname, 0))
                    if k <= 0:
                        continue
                    score = m.weight.detach().abs().sum(dim=1)  # [out_features]
                    keep = min(k, score.numel())
                    topk_idx = torch.topk(score, keep, largest=True).indices
                    prune_indices[lname] = topk_idx
                    print(f"[mag_tp]   {lname}: keeping {keep}/{score.numel()} neurons")
            
            print(f"[mag_tp] Computed indices for {len(prune_indices)} layers")
            
        # Step 3: Apply structural pruning
        print("[mag_tp] Applying structural pruning...")
        device_ = next(p for p in model.parameters() if p.is_floating_point()).device
        nst = NeuroselectiveTransformer5(
            model=model, 
            active_neurons=prune_indices,
            layer_name_map=layer_map, 
            tune_pruned=False,
            device=device_, 
            verbose=True
        )
        pruned_model = nst.transform().to(device_)
        
        # Update model reference
        model = pruned_model
        trainer.model = model  # Update trainer's model reference
        
        # Freeze all parameters for one-shot evaluation
        for p in model.parameters():
            p.requires_grad = False
            
        print(f"[mag_tp] Pruned model params: {sum(p.numel() for p in model.parameters()):,}, "
              f"trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Step 4: One-shot evaluation on pruned model (before recovery)
        print("\n[mag_tp] === PHASE 2: One-shot Evaluation on Pruned Model ===")
        pruned_run_name = run_name + "_pruned_oneshot"
        accuracy_pruned, predictions_pruned = trainer.test(
            fname=os.path.join(args.output_dir, pruned_run_name),
            task=args.task,
            eval_dataset=test_dataset,
            model_name=args.model_name,
            apply_chat_template=args.apply_chat_template,
        )
        
        # Measure pruned model inference FLOPs
        eval_dataloader_pruned = DataLoader(test_dataset, batch_size=args.eval_batch, shuffle=False)
        inf_pruned = estimate_flops_infer(
            model=model,
            data=eval_dataloader_pruned,
            modality="llm",
            tokenizer=tokenizer,
            exclude_embeddings=True
        )
        total_eval_flops_pruned = inf_pruned["flops"]
        N_eval_pruned = max(inf_pruned["N"], 1)
        
        print(f"[FLOPs][pruned-oneshot] Inference ≈ {total_eval_flops_pruned/1e15:.3f} PFLOPs "
              f"({total_eval_flops_pruned/1e12:.3f} TFLOPs); "
              f"per-example ≈ {(total_eval_flops_pruned/N_eval_pruned)/1e9:.3f} GFLOPs")
        print(f"[mag_tp][pruned-oneshot] Accuracy: {accuracy_pruned:.4f}, "
              f"Predictions: {len(predictions_pruned)}")
        
        if args.use_wandb:
            wandb.log({
                "eval_pruned_oneshot/accuracy": accuracy_pruned,
                "eval_pruned_oneshot/num_predictions": len(predictions_pruned),
                "flops/eval_pruned_oneshot_total": total_eval_flops_pruned,
                "flops/eval_pruned_oneshot_total_pflops": total_eval_flops_pruned/1e15,
                "flops/eval_pruned_oneshot_total_tflops": total_eval_flops_pruned/1e12,
                "flops/eval_pruned_oneshot_per_example_gflops": (total_eval_flops_pruned/N_eval_pruned)/1e9,
            })
        
        # ===== ENHANCED MEMORY CLEANUP BEFORE RECOVERY =====
        print("\n[mag_tp] === Memory Cleanup Before Recovery ===")
        
        # Clear evaluation objects
        try:
            del eval_dataloader_pruned, inf_pruned, predictions_pruned, accuracy_pruned
        except Exception as e:
            print(f"[mag_tp] Warning: cleanup of eval objects failed: {e}")
        
        # Clear original trainer state thoroughly
        try:
            if hasattr(trainer, "optimizer") and trainer.optimizer is not None:
                trainer.optimizer.state.clear()
                del trainer.optimizer
                trainer.optimizer = None
            
            if hasattr(trainer, "lr_scheduler") and trainer.lr_scheduler is not None:
                del trainer.lr_scheduler
                trainer.lr_scheduler = None
                
            # Clear trainer caches
            if hasattr(trainer, 'state'):
                trainer.state = None
            if hasattr(trainer, 'train_dataloader'):
                trainer.train_dataloader = None
            if hasattr(trainer, 'eval_dataloader'):
                trainer.eval_dataloader = None
                
            del trainer
        except Exception as e:
            print(f"[mag_tp] Warning: trainer cleanup failed: {e}")
        
        # Clear original optimizer if exists
        try:
            if 'optimizer' in locals() and optimizer is not None:
                optimizer.state.clear()
                del optimizer
        except Exception as e:
            print(f"[mag_tp] Warning: optimizer cleanup failed: {e}")
        
        # Force garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            print(f"[mag_tp] GPU memory after cleanup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        # ===== RECOVERY TRAINING PHASE =====
        print("\n[mag_tp] === PHASE 3: Recovery Fine-tuning ===")
        
        # Enable gradients only for pruned layers and output head
        trainable_params_list = []
        for name, param in model.named_parameters():
            param.requires_grad = False
            
        for name, module in model.named_modules():
            if isinstance(module, NeuroselectiveLinear) or ('lm_head' in name):
                for param in module.parameters():
                    param.requires_grad = True
                    trainable_params_list.append(param)
        
        if not trainable_params_list:
            raise RuntimeError("[mag_tp] No trainable parameters found for recovery")
        
        print(f"[mag_tp][recovery] Trainable parameters: {sum(p.numel() for p in trainable_params_list):,}")
        
        # Create optimizer with memory-efficient settings
        optimizer_recovery = AdamW(
            trainable_params_list, 
            lr=args.recovery_lr,  # Use recovery_lr instead of lr
            weight_decay=args.wd,
            foreach=False  # Disable multi-tensor operations to save memory
        )

        # Configure training args with memory optimization
        recovery_args = SFTConfig(
            output_dir=os.path.join(args.output_dir, run_name + "_pruned_recovery"),
            num_train_epochs=1,
            per_device_train_batch_size=max(1, args.train_batch // 2),  # Reduce batch size
            per_device_eval_batch_size=args.eval_batch,
            seed=args.seed,
            eval_strategy="no",
            save_strategy="no",
            lr_scheduler_type="linear" if args.schedule else "constant",
            load_best_model_at_end=False,
            max_length=256,
            dataset_text_field="text",
            bf16=False,
            warmup_ratio=0.0,
            learning_rate=args.recovery_lr,
            weight_decay=args.wd,
            max_grad_norm=0.5,
            gradient_accumulation_steps=args.gradient_accumulation_steps * 2, 
            dataloader_pin_memory=False, 
            deepspeed=(args.deepspeed if use_deepspeed else None),
        )

        # Create recovery trainer
        recovery_trainer = CustomSFTTrainerV2(
            task=args.task,
            model=model,
            args=recovery_args,
            train_dataset=train_dataset,
            eval_dataset=None,
            processing_class=tokenizer,
            data_collator=data_collator,
            optimizers=(optimizer_recovery, None) if (not use_deepspeed) else (None, None)
        )

        # Optional: Estimate FLOPs for recovery training
        try:
            train_dataloader_rec = DataLoader(train_dataset, batch_size=recovery_args.per_device_train_batch_size, shuffle=False)
            tr_stats_rec = estimate_flops_train(
                model=model, data=train_dataloader_rec, modality="llm",
                epochs=1, tokenizer=tokenizer
            )
            print(f"[FLOPs][recovery] Train ≈ {tr_stats_rec['flops']/1e15:.3f} PFLOPs "
                f"({tr_stats_rec['flops']/1e12:.3f} TFLOPs); "
                f"N={tr_stats_rec['N']:,}, MAX_TOKEN=2048, epochs=1")
            if args.use_wandb:
                wandb.log({
                    "flops/train_recovery_total": tr_stats_rec["flops"],
                    "flops/train_recovery_total_pflops": tr_stats_rec["flops"]/1e15,
                    "flops/train_recovery_total_tflops": tr_stats_rec["flops"]/1e12,
                })
        except Exception as e:
            logging.warning(f"[FLOPs][recovery] Train estimation failed: {e}")

        # Start recovery training
        if args.use_wandb:
            wandb.log({"status": "recovery_training_started"})
        
        recovery_trainer.train()
        
        if args.use_wandb:
            wandb.log({"status": "recovery_training_finished"})

        pruned_rec_run_name = run_name + "_pruned_recovery"
        accuracy_pruned_rec, predictions_pruned_rec = recovery_trainer.test(
            fname=os.path.join(args.output_dir, pruned_rec_run_name),
            task=args.task,
            eval_dataset=test_dataset,
            model_name=args.model_name,
            apply_chat_template=args.apply_chat_template,
        )
        eval_dataloader_pruned_rec = DataLoader(test_dataset, batch_size=args.eval_batch, shuffle=False)
        inf_unmerged_pruned_rec = estimate_flops_infer(
            model=model, data=eval_dataloader_pruned_rec, modality="llm",
            tokenizer=tokenizer, exclude_embeddings=True
        )
        total_eval_flops_pruned_rec = inf_unmerged_pruned_rec["flops"]
        N_eval_pruned_rec = max(inf_unmerged_pruned_rec["N"], 1)
        print(f"[FLOPs][pruned+recovery] Inference (eval, unmerged) ≈ "
              f"{total_eval_flops_pruned_rec/1e15:.3f} PFLOPs "
              f"({total_eval_flops_pruned_rec/1e12:.3f} TFLOPs); "
              f"per-example ≈ {(total_eval_flops_pruned_rec/N_eval_pruned_rec)/1e9:.3f} GFLOPs "
              f"(L=2048, N={inf_unmerged_pruned_rec['N']:,})")
        print(f"[mag_tp][recovery] accuracy={accuracy_pruned_rec:.4f}, "
              f"num_predictions={len(predictions_pruned_rec)}")
        if args.use_wandb:
            wandb.log({
                "eval_pruned_recovery/accuracy": accuracy_pruned_rec,
                "eval_pruned_recovery/num_predictions": len(predictions_pruned_rec),
                "flops/eval_pruned_recovery_total": total_eval_flops_pruned_rec,
                "flops/eval_pruned_recovery_total_pflops": total_eval_flops_pruned_rec/1e15,
                "flops/eval_pruned_recovery_total_tflops": total_eval_flops_pruned_rec/1e12,
                "flops/eval_pruned_recovery_per_example_gflops": (total_eval_flops_pruned_rec/N_eval_pruned_rec)/1e9,
                "tokens/eval_pruned_recovery_D": inf_unmerged_pruned_rec["D"],
                "tokens/eval_pruned_recovery_L": 2048,
                "size/eval_pruned_recovery_N": inf_unmerged_pruned_rec["N"],
            })

    if args.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()