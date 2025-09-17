import argparse
import os
import random
from peft import TaskType
import torch
from trl import DataCollatorForCompletionOnlyLM, SFTConfig
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from collections import defaultdict
from fvcore.nn import FlopCountAnalysis
import torch
from diet.mlps.ns_linear import NeuroselectiveLinear
from diet.ns_lora import NSLoraLinear
# from diet.tracker2 import NeuronTracker
from diet.tracker6 import NeuronTracker6 as NeuronTracker

# Import from peft for LoRA
from peft import get_peft_model, LoraConfig, AdaLoraConfig, LoHaConfig, LoKrConfig
import re
import logging
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support 
import math
from typing import Iterable, Optional, Union


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_model_layers(model):
    """Print layer structure of the model for debugging."""
    print("\nModel Layer Structure:")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"{name}: {module}")
    print("\n")


def adapt_neuron_indices_for_transformer(tracker_indices, model):
    """
    Adapts indices from NeuronTracker format to NeuroselectiveModelTransformer format.

    Args:
        tracker_indices: Dictionary of indices from NeuronTracker
        model: The model being transformed

    Returns:
        Dictionary with keys properly formatted for NeuroselectiveModelTransformer
    """
    transformer_indices = {}

    # Get all linear layer names from the model
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append(name)

    # Convert NeuronTracker keys to transformer keys
    for tracker_key, indices in tracker_indices.items():
        # Check if this is a layer_X_mlp_Y format key
        parts = tracker_key.split('_')
        if len(parts) >= 4 and parts[0] == 'layer' and 'mlp' in parts:
            layer_idx = parts[1]
            proj_type = parts[3]  # The projection type (e.g., gate_proj, intermediate.dense)

            # Find matching layer in the model
            matches = []
            for layer_name in linear_layers:
                # Check if the layer has both the index and projection type
                if f".{layer_idx}." in layer_name and proj_type in layer_name:
                    matches.append(layer_name)
                # Alternative format with brackets
                elif f"[{layer_idx}]" in layer_name and proj_type in layer_name:
                    matches.append(layer_name)

            if matches:
                # If multiple matches, take the shortest (most specific) one
                matches.sort(key=len)
                model_layer_name = matches[0]
                transformer_key = f"{model_layer_name}_out"  # Typically we want to prune outputs
                transformer_indices[transformer_key] = indices
                logging.info(f"Mapped {tracker_key} to {transformer_key}")
            else:
                # If no exact match, try a more flexible approach
                for layer_name in linear_layers:
                    if proj_type in layer_name:
                        # Check if there's a number in the layer name that matches our layer index
                        if re.search(r'\.{}\.'.format(layer_idx), layer_name) or \
                                re.search(r'\[{}\]'.format(layer_idx), layer_name):
                            transformer_key = f"{layer_name}_out"
                            transformer_indices[transformer_key] = indices
                            logging.info(f"Flexibly mapped {tracker_key} to {transformer_key}")
                            break
        else:
            # For non-standard keys, try direct mapping if possible
            for layer_name in linear_layers:
                if tracker_key in layer_name:
                    transformer_key = f"{layer_name}_out"
                    transformer_indices[transformer_key] = indices
                    logging.info(f"Direct mapped {tracker_key} to {transformer_key}")
                    break

    logging.info(f"Adapted {len(transformer_indices)}/{len(tracker_indices)} indices for transformer")
    return transformer_indices


class VisionEncoderWithClassifier(nn.Module):
    """Wrapper for vision encoders with classification head."""

    def __init__(self, vision_encoder, num_classes):
        super().__init__()
        self.vision_encoder = vision_encoder
        ##### TODO: Debug Config
        self.config = vision_encoder.config
        if hasattr(vision_encoder.config, 'hidden_size'):
            hidden_size = vision_encoder.config.hidden_size
        elif hasattr(vision_encoder.config, 'projection_dim'):
            hidden_size = vision_encoder.config.projection_dim
        elif hasattr(vision_encoder.config, 'vision_config') and hasattr(vision_encoder.config.vision_config,
                                                                         'hidden_size'):
            hidden_size = vision_encoder.config.vision_config.hidden_size
        else:
            hidden_size = 768

        # Add classifier head
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values, **kwargs):
        if hasattr(self.vision_encoder, 'get_image_features'):
            features = self.vision_encoder.get_image_features(pixel_values)
        else:
            outputs = self.vision_encoder(pixel_values, **kwargs)
            if hasattr(outputs, 'pooler_output'):
                features = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                features = outputs.last_hidden_state.mean(dim=1)
            else:
                features = outputs[0][:, 0]

        logits = self.classifier(features)

        return type('obj', (object,), {'logits': logits})


def evaluate_classification(model, eval_dataloader, device, modality="image", description="Evaluating",
                            cola=False, f1=False, stsb=False):
    """Evaluate the model on a validation or test set.

    Args:
        model: Model to evaluate.
        eval_dataloader: DataLoader with evaluation data.
        device: Device to use.
        modality: Data modality ("image" or "text").
        description: Description for the progress bar.
        cola: Flag indicating whether to use the CoLA evaluation metric (Matthews correlation coefficient).
        f1: Flag indicating whether to report F1 score (weighted average). Ignored if stsb or cola is True.
        stsb: Flag indicating whether to use Spearman correlation for the STS-B dataset (regression-based evaluation).

    Returns:
        tuple: (metric, avg_loss, throughput, stats)
            - metric:
                * For STS-B (stsb=True): Spearman correlation.
                * For CoLA (cola=True): Matthews correlation coefficient.
                * For F1 (f1=True): Weighted F1 score.
                * Otherwise: Accuracy.
            - avg_loss: Average loss per batch.
            - throughput: Average samples processed per second.
            - stats: Dictionary with detailed timing statistics.
    """

    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    total_time = 0
    total_samples = 0

    # Determine whether to accumulate predictions and labels based on flags.
    # Note: stsb takes precedence over both cola and f1.
    use_accumulated_metrics = stsb or cola or f1
    all_preds = [] if use_accumulated_metrics else None
    all_labels = [] if use_accumulated_metrics else None

    # Track detailed timing stats
    timing_stats = {
        'batch_times': [],
        'samples_per_batch': [],
        'batch_throughputs': []
    }

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc=description):
            # Create CUDA events for timing (if using a CUDA device)
            batch_start = torch.cuda.Event(enable_timing=True)
            batch_end = torch.cuda.Event(enable_timing=True)

            if modality == "image":
                # Handle image modality
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                batch_size = labels.size(0)

                # Time the forward pass
                batch_start.record()
                outputs = model(images)
                batch_end.record()

                # Extract logits whether outputs has a logits attribute or is a raw tensor.
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            elif modality == "text":
                # Handle text modality (assumes batch is a tuple with inputs, attention mask, and labels)
                inputs, attention_mask, labels = batch[0], batch[1], batch[-1]
                inputs = inputs.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                batch_size = labels.size(0)

                # Time the forward pass
                batch_start.record()
                outputs = model(inputs, attention_mask=attention_mask)
                batch_end.record()

                logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            else:
                raise ValueError("Unsupported modality type. Choose 'text' or 'image'.")

            # Synchronize CUDA and measure time
            torch.cuda.synchronize()
            batch_time = batch_start.elapsed_time(batch_end) / 1000.0  # Convert ms to seconds

            # Compute loss and collect predictions/labels
            if stsb:
                # For STS-B, treat the task as regression.
                # Assume model outputs one value per example; squeeze extra dims if necessary.
                preds = logits.squeeze()
                # Compute MSE loss; labels should be floats.
                loss = nn.MSELoss()(preds, labels.float())
                total_loss += loss.item()

                # Accumulate predictions and ground truth
                all_preds.extend(preds.detach().cpu().numpy().tolist())
                all_labels.extend(labels.detach().cpu().numpy().tolist())
            else:
                # For classification tasks.
                loss = nn.CrossEntropyLoss()(logits, labels)
                total_loss += loss.item()

                predictions = logits.argmax(dim=1)

                if cola or f1:
                    all_preds.extend(predictions.detach().cpu().numpy().tolist())
                    all_labels.extend(labels.detach().cpu().numpy().tolist())
                else:
                    correct += (predictions == labels).sum().item()

            total += batch_size

            # Track throughput timing
            total_time += batch_time
            total_samples += batch_size

            # Record detailed timing stats for this batch
            timing_stats['batch_times'].append(batch_time)
            timing_stats['samples_per_batch'].append(batch_size)
            timing_stats['batch_throughputs'].append(batch_size / batch_time if batch_time > 0 else 0)

    # Compute the metric based on flags:
    # Priority: stsb > cola > f1 > accuracy.
    if stsb:
        metric, _ = spearmanr(all_labels, all_preds)
    elif cola:
        metric = matthews_corrcoef(all_labels, all_preds)
    elif f1:
        metric = f1_score(all_labels, all_preds, average='weighted')
    else:
        metric = correct / total

    avg_loss = total_loss / len(eval_dataloader)
    throughput = total_samples / total_time if total_time > 0 else 0

    # Compile detailed timing statistics.
    stats = {
        'total_time': total_time,
        'total_samples': total_samples,
        'avg_batch_time': np.mean(timing_stats['batch_times']),
        'std_batch_time': np.std(timing_stats['batch_times']),
        'min_batch_time': np.min(timing_stats['batch_times']),
        'max_batch_time': np.max(timing_stats['batch_times']),
        'avg_batch_size': np.mean(timing_stats['samples_per_batch']),
        'avg_throughput': throughput,
        'p50_throughput': np.median(timing_stats['batch_throughputs']),
        'p90_throughput': np.percentile(timing_stats['batch_throughputs'], 90),
        'p95_throughput': np.percentile(timing_stats['batch_throughputs'], 95),
        'min_throughput': np.min(timing_stats['batch_throughputs']),
        'max_throughput': np.max(timing_stats['batch_throughputs'])
    }

    return metric, avg_loss, throughput, stats


def sample_active_set(dataloader, ratio=0.1, samples_per_class=None, seed = 42):
    """
    Sample a subset of the data for neuron activation analysis.

    Args:
        dataloader: PyTorch DataLoader instance
        ratio: Fraction of data to sample
        samples_per_class: Number of samples per class (if None, determined by ratio)

    Returns:
        A new DataLoader with the sampled data
    """
    # Initialize dictionary to store indices by class    
    rng = random.Random(42)   
    indices_by_class = defaultdict(list)

    print("Analyzing dataset to find samples per class...")
    dataset = dataloader.dataset
    indices_by_class = defaultdict(list)

    print("Analyzing dataset to find samples per class...")

    scan_loader = DataLoader(
        dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,
        collate_fn=getattr(dataloader, 'collate_fn', None)
    )

    # Iterate through the dataset to find examples of each class
    for idx, batch in enumerate(tqdm(scan_loader)):
        # Handle different batch structures
        if isinstance(batch, (list, tuple)):
            # Common case: (inputs, labels) or (inputs, attention_mask, labels)
            if len(batch) == 2:  # (inputs, labels)
                labels = batch[1]
            elif len(batch) == 3:  # (inputs, attention_mask, labels)
                labels = batch[2]
            elif len(batch) == 4:  # (inputs, attention_mask, token_type_ids, labels)
                labels = batch[3]
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements")
        else:
            labels = batch['labels']

        if isinstance(labels, torch.Tensor):
            batch_labels = labels.cpu().numpy()
        else:
            batch_labels = labels

        batch_size = len(batch_labels)
        for i in range(batch_size):
            label = batch_labels[i]
            sample_idx = idx * scan_loader.batch_size + i
            indices_by_class[label].append(sample_idx)

    if samples_per_class is None:
        # Calculate samples per class based on ratio
        total_samples = sum(len(indices) for indices in indices_by_class.values())
        num_classes = len(indices_by_class)
        samples_per_class = int(total_samples * ratio / num_classes)

    sampled_indices = []
    print(f"Found {len(indices_by_class)} classes")
    for label, indices in indices_by_class.items():
        n_samples = min(samples_per_class, len(indices))
        print(f"Class {label}: Found {len(indices)} samples, taking {n_samples}")
        sampled_indices.extend(rng.sample(indices, n_samples))

    subset_dataset = Subset(dataloader.dataset, sampled_indices)

    sampled_dataloader = DataLoader(
        subset_dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,
        collate_fn=dataloader.collate_fn if hasattr(dataloader, 'collate_fn') else None
    )

    print(f"Created active sampled dataloader with {len(subset_dataset)} total samples")
    return sampled_dataloader



def track_neuron_activations(model, dataloader, device, threshold=0, tokenizer=None, use_abs=False,
                            track_attention_proj=True, attention_proj_patterns=None, verbose=False):
    # print(f"Tracking neuron activations with threshold={threshold}, use_abs={use_abs}, track_attn_proj={track_attention_proj}")

    tracker = NeuronTracker(
        model=model,
        threshold=threshold,
        use_abs_threshold=use_abs,
        device=device,
        tokenizer=tokenizer,
        track_attention_proj=track_attention_proj,
        # attention_proj_patterns=attention_proj_patterns,
        verbose=verbose
    )


    return tracker.get_active_indices(dataloader)


def setup_ns_lora(model, lora_config):
    """
    Apply NSLoraLinear wrapper to NeuroselectiveLinear layers.

    Args:
        model: Model to adapt
        lora_config: Configuration with r, alpha, etc.

    Returns:
        Model with NSLoraLinear wrappers applied
    """
    from diet.ns_lora import NSLoraLinear

    # Wrap NeuroselectiveLinear layers with NSLoraLinear
    for name, module in model.named_modules():
        if isinstance(module, NeuroselectiveLinear):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model
            for part in parent_name.split('.'):
                if part:
                    parent = getattr(parent, part)

            # Create NSLoraLinear wrapper
            ns_lora = NSLoraLinear(
                ns_linear=module,
                r=lora_config['r'],
                lora_alpha=lora_config['alpha'],
                lora_dropout=lora_config['dropout'],
            )
            setattr(parent, child_name, ns_lora)
            print(f"Applied NSLoraLinear to {name}")

    return model


def setup_lora(model, lora_config, attn=False, ns=False):
    """
    Apply LoRA to the model.

    Args:
        model: Model to adapt
        lora_config: Configuration with r, alpha, etc.
        attn: Whether to apply LoRA to attention layers
        ns: Whether to apply NSLoraLinear to NeuroselectiveLinear layers

    Returns:
        Model with LoRA applied
    """
    # First apply NSLoraLinear if requested
    if ns:
        model = setup_ns_lora(model, lora_config)

    # Get target modules for regular LoRA
    exact_target_modules = []
    for name, module in model.named_modules():
        if not attn and "attn" in name or "attention" in name:
            continue
        if isinstance(module, nn.Linear):
            if ns and isinstance(module, (NeuroselectiveLinear, NSLoraLinear)):
                continue
            exact_target_modules.append(name)

    exact_target_modules = sorted(set(exact_target_modules))
    print("Exact target modules for LoRA:", exact_target_modules)

    if lora_config["type"] == "lora":
        config = LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["alpha"],
            lora_dropout=lora_config["dropout"],
            target_modules=exact_target_modules
            # task_type = TaskType.CAUSAL_LM  
        )
    elif lora_config["type"] == "adalora":
        config = AdaLoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["alpha"],
            lora_dropout=lora_config["dropout"],
            beta1=0.85,
            beta2=0.85,
            tinit=200,
            deltaT=10,
            target_modules=exact_target_modules
        )
    elif lora_config["type"] == "loha":
        config = LoHaConfig(
            r=lora_config["r"],
            alpha=lora_config["alpha"],
            dropout=lora_config["dropout"],
            target_modules=exact_target_modules
        )
    elif lora_config["type"] == "lokr":
        config = LoKrConfig(
            r=lora_config["r"],
            alpha=lora_config["alpha"],
            dropout=lora_config["dropout"],
            target_modules=exact_target_modules
        )
    else:
        raise ValueError(f"Unknown LoRA type: {lora_config['type']}")
    # Apply LoRA
    model = get_peft_model(model, config)
    model.config.return_dict = True  
    # Print LoRA-applied modules
    print("\nLoRA applied to the following modules:")
    for name, module in model.named_modules():
        if hasattr(module, "lora_A"):  # LoRA-modified layers have lora_A parameter
            print(f"{name}")

    return model


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_jaccard_similarity(dict1, dict2):
    """Calculates the average Jaccard similarity between two active_indices dictionaries."""
    all_keys = set(dict1.keys()) & set(dict2.keys())
    if '_attn_proj_layers' in all_keys:
        all_keys.remove('_attn_proj_layers')
        
    similarities = []
    
    for key in all_keys:
        set1 = set(dict1[key].cpu().numpy())
        set2 = set(dict2[key].cpu().numpy())
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            similarities.append(1.0)
        else:
            similarities.append(intersection / union)
            
    if not similarities:
        return 0.0
        
    return sum(similarities) / len(similarities)

def calculate_directed_coverage(dictA, dictB, weighted=True):
    """Cov(A|B): layerwise intersection-over-A, then aggregate.
       Args:
         dictA, dictB: {layer_name: tensor/list/ndarray of indices}
         weighted: True -> weight by |A_layer|; False -> unweighted mean
    """
    keys = set(dictA.keys()) & set(dictB.keys())
    if '_attn_proj_layers' in keys:
        keys.remove('_attn_proj_layers')

    num, den, count = 0.0, 0.0, 0
    for k in keys:
        # to set[int]
        A = dictA[k]
        B = dictB[k]
        try:
            import torch
            if hasattr(A, "detach"): A = A.detach().cpu().numpy()
            if hasattr(B, "detach"): B = B.detach().cpu().numpy()
        except Exception:
            pass
        import numpy as np
        A = set(int(i) for i in np.array(A).ravel().tolist())
        B = set(int(i) for i in np.array(B).ravel().tolist())

        kA = len(A)
        inter = len(A & B)

        if kA == 0:
            cov_l = 1.0   
            w = 0.0 if weighted else 1.0
        else:
            cov_l = inter / kA
            w = float(kA) if weighted else 1.0

        num += cov_l * w
        den += w
        count += 1

    if count == 0:
        return 1.0  
    if den == 0:
        return 1.0
    return float(num / den)
