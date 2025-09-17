import torch
import torch.nn as nn
import logging
from typing import Dict, Optional, List, Any, Union

logger = logging.getLogger(__name__)

# From our NeuroselectiveLinear implementation
from .mlps.ns_linear2 import NeuroselectiveLinear

# ===== Add for name matching ===
def norm_key(s: str) -> str:
    return s.replace('.', '_').replace('-', '_')

def join_norm(*parts: str) -> str:
    return norm_key(".".join(parts))


# Common MLP submodule names
MLP_SUBMODULE_NAMES = {
    "gate": ["gate_proj", "fc1", "w1", "wi_0", "intermediate.dense","lin1"],
    "up": ["up_proj", "w2", "wi_1"],
    "down": ["down_proj", "fc2", "wo", "output.dense","lin2"] 
}

class NeuroselectiveTransformer5:
    def __init__(
            self,
            model: nn.Module,
            active_neurons: Dict[str, Union[torch.Tensor, List[int]]],
            layer_name_map: Optional[Dict[nn.Module, str]] = None,
            device: Optional[Union[str, torch.device]] = None,
            verbose: bool = False,
            tune_pruned: bool = False
    ):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.tune_pruned = tune_pruned
        self.layer_name_map = layer_name_map

        # Convert active neuron lists to tensors
        self.active_neurons = {}
        for k, v in active_neurons.items():
            if isinstance(v, list):
                self.active_neurons[k] = torch.tensor(v, dtype=torch.long, device=self.device)
            elif isinstance(v, torch.Tensor):
                self.active_neurons[k] = v.to(self.device).long()
        # for k, v in active_neurons.items():
        #     if isinstance(v, list):
        #         # to cpu first, then to gpu
        #         self.active_neurons[k] = torch.tensor(v, dtype=torch.long, device="cpu")
        #     elif isinstance(v, torch.Tensor):
        #         self.active_neurons[k] = v.detach().long().cpu()


        self.original_params = sum(p.numel() for p in model.parameters())
        self.transformed_mlp_blocks = {}
        self.parameter_stats = {}

    def _replace_module(self,
                        parent_module: nn.Module,   # input block
                        child_name: str,            # could be either fc1 or xx.inter.dense
                        new_module: nn.Module) -> bool:
        """
        support name replace 
            "fc1" → replace
            "intermediate.dense" → find block.intermediate, then replace dense
        """
        try:
            parts = child_name.split(".")
            tgt_parent = parent_module
            for p in parts[:-1]:                    
                tgt_parent = getattr(tgt_parent, p)
            setattr(tgt_parent, parts[-1], new_module)
            return True
        except Exception as e:
            logger.error(f"Failed to replace module '{child_name}': {e}")
            return False

    def _find_key_candidates(self, block_name: str, sub_name: str, module: nn.Module) -> list:
        cands = []
        # 1) try layer_name_map
        if self.layer_name_map:
            orig = self.layer_name_map.get(module)
            if orig:
                cands.append(norm_key(orig))
        # 2)downgrade
        cands.append(join_norm(block_name, sub_name))
        # 3) some use _ rather than .
        cands.append(join_norm(block_name.replace("model.layers", "model_layers"), sub_name))
        return cands

    def _get_idx_by_candidates(self, cands: list):
        for k in cands:
            if k in self.active_neurons:
                return self.active_neurons[k], k
        return None, None
    # ===============================================================

    @staticmethod
    def _is_two_layer_ffn(module: nn.Module) -> bool:
        """
        TODO: Check 
        For BERT/RoBERTa/DistilBERT style blocks:
        <TransformerLayer>
           ├─ intermediate.dense (Linear)
           └─ output.dense       (Linear)
        """
        return (
            hasattr(module, "intermediate")
            and isinstance(getattr(module.intermediate, "dense", None), nn.Linear)
            and hasattr(module, "output")
            and isinstance(getattr(module.output, "dense", None), nn.Linear)
        )
    # ───────────────────────────────────────────────

    def _find_mlp_blocks(self) -> Dict[str, nn.Module]:
        mlp_blocks = {}
        for name, module in self.model.named_modules():

            is_ffn_by_structure = self._is_two_layer_ffn(module)

            is_ffn_by_name = (
                any(key in name.lower() for key in ["mlp", "ffn", "feed_forward"])
                and any(isinstance(ch, nn.Linear) for ch in module.children())
            )

            if is_ffn_by_structure or is_ffn_by_name:
                is_child = any(name.startswith(p + ".") for p in mlp_blocks)
                if not is_child:
                    mlp_blocks[name] = module

        logger.info(f"Found {len(mlp_blocks)} MLP blocks")
        return mlp_blocks
    # helper: safely fetch nested attribute like "intermediate.dense"
    # ----------------------------------------------------------
    def _get_submodule(self, root: nn.Module, dotted_name: str) -> nn.Module:
        mod = root
        for part in dotted_name.split("."):
            mod = getattr(mod, part)
        return mod


    def _find_mlp_submodules(self, mlp_block: nn.Module):
        """Find gate, up, and down projection layers in an MLP block."""
        # block_children = dict(mlp_block.named_children())
        # if "fc1" in block_children and "fc2" in block_children:
        #    if self.verbose:
        #         print(f"[DEBUG _find_mlp_submodules] Detected 2-layer FFN in block '{mlp_block}'")
        #    return "fc1", None, "fc2"

        # gate_name, up_name, down_name = None, None, None
        block_children = dict(mlp_block.named_children())
        if "fc1" in block_children and "fc2" in block_children:
            if self.verbose:
                print(f"[DEBUG _find_mlp_submodules] Detected ViT 2-layer FFN in '{mlp_block}'")
            return "fc1", None, "fc2"
        if "lin1" in block_children and "lin2" in block_children:
            if self.verbose:
                print(f"[DEBUG] DistilBERT FFN found in '{mlp_block}'")
            return "lin1", None, "lin2"
        if (
            hasattr(mlp_block, "intermediate")
            and isinstance(getattr(mlp_block.intermediate, "dense", None), nn.Linear)
            and hasattr(mlp_block, "output")
            and isinstance(getattr(mlp_block.output, "dense", None), nn.Linear)
        ):

            if self.verbose:
                print(f"[DEBUG _find_mlp_submodules] Detected BERT-style FFN in '{mlp_block}'")
            return "intermediate.dense", None, "output.dense"

        gate_name, up_name, down_name = None, None, None
        # Find first match for each role
        for potential_gate_name in MLP_SUBMODULE_NAMES["gate"]:
            if potential_gate_name in block_children and isinstance(block_children[potential_gate_name], nn.Linear):
                gate_name = potential_gate_name
                break

        for potential_up_name in MLP_SUBMODULE_NAMES["up"]:
            if potential_up_name in block_children and isinstance(block_children[potential_up_name], nn.Linear):
                up_name = potential_up_name
                break

        for potential_down_name in MLP_SUBMODULE_NAMES["down"]:
            if potential_down_name in block_children and isinstance(block_children[potential_down_name], nn.Linear):
                down_name = potential_down_name
                break

        return gate_name, up_name, down_name

    def _get_key_for_module(self, module: nn.Module) -> Optional[str]:
        """Find the corresponding key in active_neurons for a module."""
        if self.layer_name_map:
            original_name = self.layer_name_map.get(module)
            if original_name:
                # Try to find a matching key
                clean_name = original_name.replace(".", "_").replace("-", "_")
                if clean_name in self.active_neurons:
                    return clean_name

        return None

    def transform(self) -> nn.Module:
        """Transform the model by replacing linear layers with NS linear layers."""
        logger.info("Starting intermediate dimension pruning")
        num_blocks_processed = 0
        num_blocks_pruned = 0

        mlp_blocks = self._find_mlp_blocks()

        if not mlp_blocks:
            logger.warning("No MLP blocks found")
            return self.model

        for block_name, block_module in mlp_blocks.items():
            num_blocks_processed += 1
            gate_key = None
            intermediate_indices = None
            replace_success = False

            # print("[DEBUG transform]":block_name)
            try:
                # Find submodules
                gate_name, up_name, down_name = self._find_mlp_submodules(block_module)
                if self.verbose:
                    print(f"[DEBUG transform] Block={block_name} → gate={gate_name}, up={up_name}, down={down_name}")
                # is_two_layer_ffn = up_name is None

                # if not (gate_name and up_name and down_name):
                #     logger.warning(f"Skipping block '{block_name}': Could not identify all required submodules")
                #     continue
                is_two_layer_ffn = (up_name is None)
                # check gate & down 
                if gate_name is None or down_name is None:
                    logger.warning(f"Skipping block '{block_name}': missing gate/down layer")
                    continue

                # Get module instances
                # gate_layer = getattr(block_module, gate_name)
                # # up_layer = getattr(block_module, up_name)
                # up_layer = getattr(block_module, up_name) if not is_two_layer_ffn else None
                # down_layer = getattr(block_module, down_name)
                gate_layer = self._get_submodule(block_module, gate_name)
                up_layer   = self._get_submodule(block_module, up_name) if not is_two_layer_ffn else None
                down_layer = self._get_submodule(block_module, down_name)

                # Find active neurons
                # gate_key = self._get_key_for_module(gate_layer)
                # if self.verbose:
                #     print(f"[DEBUG transform] Block={block_name} → gate_key={gate_key}")

                # if not gate_key or gate_key not in self.active_neurons:
                #     logger.warning(f"Skipping block '{block_name}': No active neurons found for gate layer")
                #     continue

                # intermediate_indices = self.active_neurons[gate_key]
                # intermediate_indices = self.active_neurons[gate_key].to(gate_layer.weight.device).long()
                cand_keys = self._find_key_candidates(block_name, gate_name, gate_layer)
                intermediate_indices, hit_key = self._get_idx_by_candidates(cand_keys)
                gate_key = hit_key
                if self.verbose:
                    print(f"[DEBUG transform] Block={block_name} → candidates={cand_keys[:3]}... hit={hit_key}")
                if intermediate_indices is None or intermediate_indices.numel() == 0:
                    logger.warning(f"Skipping block '{block_name}': No active neurons for gate layer (tried {len(cand_keys)} keys)")
                    continue
                intermediate_indices = intermediate_indices.view(-1).to('cpu').long()
                if intermediate_indices.numel() == 0:
                    logger.warning(f"Skipping block '{block_name}': Gate layer has 0 active neurons")
                    continue

                pruned_intermediate_dim = len(intermediate_indices)
                original_intermediate_dim = gate_layer.out_features

                logger.info(f"Pruning block '{block_name}': {original_intermediate_dim} -> {pruned_intermediate_dim}")

                # Create new pruned layers
                # factory_kwargs = {
                #     'device': gate_layer.weight.device,
                #     'dtype': gate_layer.weight.dtype
                # }
                orig_device = gate_layer.weight.device
                orig_dtype  = gate_layer.weight.dtype
                factory_kwargs = {'device': orig_device, 'dtype': orig_dtype}

                is_two_layer_ffn = (up_name is None) or (not isinstance(up_layer, nn.Linear))

                # ------------- gate / fc1 -------------
                # new_gate_layer = NeuroselectiveLinear.from_linear(
                #     original_module=gate_layer,
                #     in_indices=None,                 # keep input
                #     out_indices=intermediate_indices,  # prune out
                #     **factory_kwargs
                # )

                # # ------------- up_proj / w2 -------------
                # if not is_two_layer_ffn:
                #     new_up_layer = NeuroselectiveLinear.from_linear(
                #         original_module=up_layer,
                #         in_indices=None,
                #         out_indices=intermediate_indices,
                #         **factory_kwargs
                #     )

                # # ------------- down / fc2 -------------
                # new_down_layer = NeuroselectiveLinear.from_linear(
                #     original_module=down_layer,
                #     in_indices=intermediate_indices,  # keep input
                #     out_indices=None,                 # prune out
                #     **factory_kwargs
                # )
                new_gate_layer = NeuroselectiveLinear.from_linear(
                    original_module=gate_layer, 
                    in_indices=None,
                    out_indices=intermediate_indices,
                    **factory_kwargs
                 )

                if not is_two_layer_ffn:
                    new_up_layer = NeuroselectiveLinear.from_linear(
                        original_module=up_layer,
                        in_indices=None,
                        out_indices=intermediate_indices,
                        **factory_kwargs
                     )

                new_down_layer = NeuroselectiveLinear.from_linear(
                    original_module=down_layer,
                    in_indices=intermediate_indices,
                    out_indices=None,
                    **factory_kwargs
                    )



                # Replace modules
                replace_success = True
                # if not self._replace_module(block_module, gate_name, new_gate_layer):
                #     replace_success = False
                # if not self._replace_module(block_module, up_name, new_up_layer):
                #     replace_success = False
                # if not self._replace_module(block_module, down_name, new_down_layer):
                #     replace_success = False
                if not self._replace_module(block_module, gate_name, new_gate_layer):
                    replace_success = False
                if not is_two_layer_ffn:
                    if not self._replace_module(block_module, up_name, new_up_layer):
                        replace_success = False
                if not self._replace_module(block_module, down_name, new_down_layer):
                    replace_success = False
                # if replace_success:
                #     self.transformed_mlp_blocks[block_name] = {
                #         "original_intermediate_dim": original_intermediate_dim,
                #         "pruned_intermediate_dim": pruned_intermediate_dim,
                #         "active_neuron_key": gate_key
                #     }
                #     num_blocks_pruned += 1
                if replace_success:
                    new_gate_layer.to(orig_device)
                    if not is_two_layer_ffn:
                        new_up_layer.to(orig_device)
                    new_down_layer.to(orig_device)

                    self.transformed_mlp_blocks[block_name] = {
                        "ffn_type": "two-layer" if is_two_layer_ffn else "three-layer",
                        "original_intermediate_dim": original_intermediate_dim,
                        "pruned_intermediate_dim": pruned_intermediate_dim,
                        "active_neuron_key": hit_key,  ### ← 用已绑定好的 gate_key
                    }
                    num_blocks_pruned += 1

                # if replace_success:
                #     new_gate_layer.to(orig_device)
                #     if not is_two_layer_ffn:
                #         new_up_layer.to(orig_device)
                #     new_down_layer.to(orig_device)
                #     self.transformed_mlp_blocks[block_name] = {
                #         "ffn_type": "two-layer" if is_two_layer_ffn else "three-layer",
                #         "original_intermediate_dim": original_intermediate_dim,
                #         "pruned_intermediate_dim": pruned_intermediate_dim,
                #         "active_neuron_key": gate_key
                #     }
                #     num_blocks_pruned += 1

            except Exception as e:
                logger.error(f"Error processing block '{block_name}': {e}")

        # Calculate statistics
        final_total_params = sum(p.numel() for p in self.model.parameters())
        final_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        absolute_reduction = self.original_params - final_total_params
        overall_reduction_perc = absolute_reduction / self.original_params if self.original_params > 0 else 0

        logger.info(f"MLP blocks processed: {num_blocks_processed}")
        logger.info(f"MLP blocks pruned: {num_blocks_pruned}")
        logger.info(f"Original params: {self.original_params:,}")
        logger.info(f"Final params: {final_total_params:,}")
        logger.info(f"Reduction: {overall_reduction_perc:.2%}")

        self.parameter_stats = {
            "initial_total_params": self.original_params,
            "final_total_params": final_total_params,
            "final_trainable_params": final_trainable_params,
            "absolute_model_reduction": absolute_reduction,
            "overall_model_reduction_perc": overall_reduction_perc * 100,
            "mlp_blocks_processed": num_blocks_processed,
            "mlp_blocks_pruned": num_blocks_pruned
        }

        return self.model

    def get_parameter_stats(self):
        return self.parameter_stats

    def get_transformed_mlp_blocks(self):
        return self.transformed_mlp_blocks
    #### ====== Added a helper to extract all subsets of active neurons ====== ####
    def record_original_active_indices(self) -> None:
        """Record active neuron indices in terms of the original weight matrix, without modifying model."""
        logger.info("Recording original intermediate indices only (no pruning)")
        num_blocks_processed = 0
        mlp_blocks = self._find_mlp_blocks()

        if not mlp_blocks:
            logger.warning("No MLP blocks found")
            return

        for block_name, block_module in mlp_blocks.items():
            num_blocks_processed += 1
            try:
                gate_name, up_name, down_name = self._find_mlp_submodules(block_module)

                if not (gate_name and up_name and down_name):
                    logger.warning(f"Skipping block '{block_name}': Could not identify all required submodules")
                    continue

                gate_layer = getattr(block_module, gate_name)
                gate_key = self._get_key_for_module(gate_layer)

                if not gate_key or gate_key not in self.active_neurons:
                    logger.warning(f"Skipping block '{block_name}': No active neurons found for gate layer")
                    continue

                intermediate_indices = self.active_neurons[gate_key]

                if intermediate_indices.numel() == 0:
                    logger.warning(f"Skipping block '{block_name}': Gate layer has 0 active neurons")
                    continue

                self.transformed_mlp_blocks[block_name] = {
                    "active_indices_orig": intermediate_indices.cpu().tolist(),
                    "original_intermediate_dim": gate_layer.out_features,
                    "pruned_intermediate_dim": len(intermediate_indices),
                    "active_neuron_key": gate_key
                }

            except Exception as e:
                logger.error(f"Error processing block '{block_name}': {e}")

        logger.info(f"Finished recording for {num_blocks_processed} MLP blocks")
