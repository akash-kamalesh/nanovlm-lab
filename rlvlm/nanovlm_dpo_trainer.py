# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
# Adapted for nanoVLM by Akash Kamalesh.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NanoVLM-compatible DPO Trainer

This module adapts the HuggingFace TRL DPOTrainer to work with nanoVLM models.
Adapted by Akash Kamalesh.

Key differences from the standard DPOTrainer:
1. Uses nanoVLM's forward signature: forward(input_ids, images, attention_mask, targets)
2. Custom data collator for nanoVLM's image format
3. Uses nanoVLM's image processor and tokenizer
4. Handles nanoVLM's logits output format
"""

import copy
import random
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import tqdm
from datasets import Dataset, IterableDataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput

# nanoVLM imports
from nanovlm.models.vision_language_model import VisionLanguageModel
from nanovlm.data.processors import get_tokenizer, get_image_processor, get_image_string
from nanovlm.models.config import VLMConfig


__all__ = [
    "NanoVLMDPOTrainer",
    "NanoVLMDataCollatorForPreference",
    "NanoVLMPreferenceDataset",
    "create_nanovlm_dpo_dataset",
    "pad",
    "pad_to_length",
    "selective_log_softmax",
    "flush_left",
]


def pad(tensors: list[torch.Tensor], padding_value: int = 0, padding_side: str = "right") -> torch.Tensor:
    """Pad a list of tensors to the same length."""
    max_len = max(t.size(0) for t in tensors)
    padded = []
    for t in tensors:
        if padding_side == "right":
            padded.append(F.pad(t, (0, max_len - t.size(0)), value=padding_value))
        else:  # left padding
            padded.append(F.pad(t, (max_len - t.size(0), 0), value=padding_value))
    return torch.stack(padded)


def pad_to_length(tensor: torch.Tensor, length: int, pad_value: int, dim: int = -1) -> torch.Tensor:
    """Pad tensor to specified length along given dimension."""
    if tensor.size(dim) >= length:
        return tensor
    pad_size = length - tensor.size(dim)
    pad_shape = [0] * (2 * tensor.dim())
    pad_shape[-(2 * dim + 1) - 1] = pad_size  # Pad on the right side of the specified dim
    return F.pad(tensor, pad_shape, value=pad_value)


def selective_log_softmax(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute log softmax and gather log probabilities for labels."""
    log_probs = F.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def flush_left(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Flush tensors to the left by removing leading padding."""
    # Find the first non-zero position for each row in the first tensor (attention mask)
    attention_mask = tensors[0]
    batch_size, seq_len = attention_mask.shape
    
    # Find first non-zero position per row
    first_nonzero = (attention_mask != 0).int().argmax(dim=1)
    
    # Create output tensors
    outputs = []
    for tensor in tensors:
        output = torch.zeros_like(tensor)
        for i in range(batch_size):
            start = first_nonzero[i].item()
            length = seq_len - start
            output[i, :length] = tensor[i, start:]
        outputs.append(output)
    
    return tuple(outputs)


@dataclass
class NanoVLMDataCollatorForPreference(DataCollatorMixin):
    """
    Data collator for nanoVLM preference data.
    
    Handles:
    - Padding of input_ids and attention masks
    - Batching of processed images (list of tensors)
    - Separate handling of chosen and rejected completions
    
    Args:
        tokenizer: The nanoVLM tokenizer
        pad_token_id: Token ID to use for padding
        return_tensors: Type of tensor to return (default "pt")
    """
    tokenizer: PreTrainedTokenizerBase
    pad_token_id: int
    return_tensors: str = "pt"
    
    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        # Convert to tensors
        prompt_input_ids = [torch.tensor(ex["prompt_input_ids"]) for ex in examples]
        prompt_attention_mask = [torch.ones_like(ids) for ids in prompt_input_ids]
        chosen_input_ids = [torch.tensor(ex["chosen_input_ids"]) for ex in examples]
        chosen_attention_mask = [torch.ones_like(ids) for ids in chosen_input_ids]
        rejected_input_ids = [torch.tensor(ex["rejected_input_ids"]) for ex in examples]
        rejected_attention_mask = [torch.ones_like(ids) for ids in rejected_input_ids]
        
        # Pad sequences
        output = {
            "prompt_input_ids": pad(prompt_input_ids, padding_value=self.pad_token_id, padding_side="left"),
            "prompt_attention_mask": pad(prompt_attention_mask, padding_value=0, padding_side="left"),
            "chosen_input_ids": pad(chosen_input_ids, padding_value=self.pad_token_id, padding_side="right"),
            "chosen_attention_mask": pad(chosen_attention_mask, padding_value=0, padding_side="right"),
            "rejected_input_ids": pad(rejected_input_ids, padding_value=self.pad_token_id, padding_side="right"),
            "rejected_attention_mask": pad(rejected_attention_mask, padding_value=0, padding_side="right"),
        }
        
        # Handle images - keep as list of tensors for nanoVLM
        if "images" in examples[0] and examples[0]["images"] is not None:
            # Each example's images is a tensor [num_patches, C, H, W]
            # Keep as list for nanoVLM's _process_images method
            output["images"] = [ex["images"] for ex in examples]
        
        # Handle precomputed reference log probs
        if "ref_chosen_logps" in examples[0] and "ref_rejected_logps" in examples[0]:
            output["ref_chosen_logps"] = torch.tensor([ex["ref_chosen_logps"] for ex in examples])
            output["ref_rejected_logps"] = torch.tensor([ex["ref_rejected_logps"] for ex in examples])
        
        # Handle image ratios for get_image_string
        if "image_ratios" in examples[0]:
            output["image_ratios"] = [ex["image_ratios"] for ex in examples]
        
        return output


class NanoVLMDPOTrainer(Trainer):
    """
    DPO Trainer adapted for nanoVLM models.
    
    This trainer implements Direct Preference Optimization (DPO) for vision-language
    models using the nanoVLM architecture.
    
    Args:
        model: The nanoVLM model to train
        ref_model: Reference model for DPO (if None, a copy of model is created)
        args: Training arguments
        train_dataset: Training dataset with preference pairs
        eval_dataset: Evaluation dataset
        tokenizer: nanoVLM tokenizer
        image_processor: nanoVLM image processor
        data_collator: Data collator (default: NanoVLMDataCollatorForPreference)
        beta: DPO beta parameter (temperature)
        loss_type: Type of DPO loss ("sigmoid", "hinge", "ipo", etc.)
        label_smoothing: Label smoothing parameter
        max_length: Maximum sequence length
        max_prompt_length: Maximum prompt length
        max_completion_length: Maximum completion length
        precompute_ref_log_probs: Whether to precompute reference log probs
    """
    
    def __init__(
        self,
        model: VisionLanguageModel,
        ref_model: Optional[VisionLanguageModel] = None,
        args=None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        image_processor: Optional[Callable] = None,
        data_collator: Optional[DataCollatorMixin] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple = (None, None),
        beta: float = 0.1,
        loss_type: str = "sigmoid",
        label_smoothing: float = 0.0,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_completion_length: Optional[int] = None,
        precompute_ref_log_probs: bool = False,
        reference_free: bool = False,
        disable_dropout: bool = True,
    ):
        # Get tokenizer from model if not provided
        if tokenizer is None:
            tokenizer = model.tokenizer
        
        # Get image processor from model config if not provided
        if image_processor is None:
            cfg = model.cfg
            resize_to_max_side_len = getattr(cfg, "resize_to_max_side_len", False)
            image_processor = get_image_processor(cfg.max_img_size, cfg.vit_img_size, resize_to_max_side_len)
        
        # Store config
        self.model_cfg = model.cfg
        self.image_processor = image_processor
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        
        # Create data collator if not provided
        if data_collator is None:
            data_collator = NanoVLMDataCollatorForPreference(
                tokenizer=tokenizer,
                pad_token_id=self.pad_token_id,
            )
        
        # DPO parameters
        self.beta = beta
        self.loss_type = loss_type if isinstance(loss_type, list) else [loss_type]
        self.label_smoothing = label_smoothing
        self.max_length = max_length or model.cfg.lm_max_length
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length
        self.precompute_ref_log_probs = precompute_ref_log_probs
        self.reference_free = reference_free
        self.label_pad_token_id = -100
        
        # Reference model
        if ref_model is None and not reference_free:
            # Create a copy of the model as reference
            self.ref_model = copy.deepcopy(model)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
        else:
            self.ref_model = ref_model
        
        if disable_dropout:
            self._disable_dropout(model)
            if self.ref_model is not None:
                self._disable_dropout(self.ref_model)
        
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        
        # Initialize parent trainer
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        
        if self.ref_model is not None:
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
    
    def _disable_dropout(self, model: nn.Module):
        """Disable dropout in the model."""
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
    
    @staticmethod
    def concatenated_inputs(
        batch: dict[str, Any],
        padding_value: int,
    ) -> dict[str, torch.Tensor]:
        """
        Concatenate chosen and rejected inputs for efficient forward pass.
        
        For nanoVLM, we need to handle:
        - input_ids: concatenate prompt + completion
        - attention_mask: concatenate masks
        - images: duplicate for chosen and rejected
        """
        output = {}
        
        output["prompt_input_ids"] = torch.cat([
            batch["prompt_input_ids"],
            batch["prompt_input_ids"]
        ], dim=0)
        output["prompt_attention_mask"] = torch.cat([
            batch["prompt_attention_mask"],
            batch["prompt_attention_mask"]
        ], dim=0)
        
        max_completion_length = max(
            batch["chosen_input_ids"].shape[1],
            batch["rejected_input_ids"].shape[1]
        )
        output["completion_input_ids"] = torch.cat([
            pad_to_length(batch["chosen_input_ids"], max_completion_length, padding_value),
            pad_to_length(batch["rejected_input_ids"], max_completion_length, padding_value),
        ], dim=0)
        output["completion_attention_mask"] = torch.cat([
            pad_to_length(batch["chosen_attention_mask"], max_completion_length, 0),
            pad_to_length(batch["rejected_attention_mask"], max_completion_length, 0),
        ], dim=0)
        
        # Handle images - duplicate for chosen and rejected
        if "images" in batch and batch["images"] is not None:
            # batch["images"] is a list of tensors, one per sample
            # Duplicate each image for chosen and rejected
            output["images"] = batch["images"] + batch["images"]
        
        return output
    
    def concatenated_forward(
        self,
        model: VisionLanguageModel,
        batch: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """
        Run forward pass on concatenated chosen and rejected inputs.
        
        Returns log probabilities for chosen and rejected completions.
        """
        num_examples = batch["prompt_input_ids"].shape[0]
        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.pad_token_id)
        
        # Concatenate prompt and completion
        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]
        
        input_ids = torch.cat([prompt_input_ids, completion_input_ids], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, completion_attention_mask], dim=1)
        
        # Create loss mask (only compute loss on completion tokens)
        loss_mask = torch.cat([
            torch.zeros_like(prompt_attention_mask),
            completion_attention_mask
        ], dim=1)
        
        # Flush left to remove leading padding BEFORE any truncation check
        attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
        
        # WARNING: Do NOT truncate input_ids if it would cut image tokens!
        # Image tokens must match the number of image patches for the vision encoder.
        # If max_length is set and sequence is too long, we skip truncation and warn.
        if self.max_length is not None and input_ids.size(1) > self.max_length:
            print(f"[WARNING] Sequence length {input_ids.size(1)} exceeds max_length {self.max_length}. "
                  f"NOT truncating to preserve image token alignment. Consider increasing max_length or filtering long samples.")
        
        # Get images
        images = concatenated_batch.get("images", None)
        
        # Forward pass through nanoVLM
        # nanoVLM forward: (input_ids, images, attention_mask, targets) -> (logits, loss)
        # When targets=None, nanoVLM returns hidden states (embeddings), not logits
        
        # Unwrap model if needed (for accelerator-wrapped models)
        if hasattr(self, 'accelerator'):
            unwrapped_model = self.accelerator.unwrap_model(model)
        elif hasattr(model, 'module'):
            unwrapped_model = model.module
        else:
            unwrapped_model = model
        
        # Count image tokens in input_ids
        image_token_id = unwrapped_model.tokenizer.image_token_id
        num_image_tokens = (input_ids == image_token_id).sum().item()
        
        hidden_states, _ = unwrapped_model(
            input_ids=input_ids,
            images=images,
            attention_mask=attention_mask,
            targets=None,  # We compute our own loss
        )
        
        # Apply LM head to get logits
        # nanoVLM's decoder.lm_use_tokens is False when used as VLM backbone,
        # so forward returns embeddings and we need to apply the head
        logits = unwrapped_model.decoder.head(hidden_states)
        
        # Compute log probabilities
        # Shift logits and labels for causal LM
        labels = torch.roll(input_ids, shifts=-1, dims=1)
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()
        
        # Set labels for non-loss positions to 0 (will be ignored)
        labels[~loss_mask] = 0
        
        # Compute per-token log probabilities
        per_token_logps = selective_log_softmax(logits[:, :-1], labels[:, :-1])
        per_token_logps[~loss_mask[:, :-1]] = 0
        
        # Sum log probabilities for each sequence
        all_logps = per_token_logps.sum(-1)
        
        # Split into chosen and rejected
        chosen_logps = all_logps[:num_examples]
        rejected_logps = all_logps[num_examples:]
        
        # Compute mean logits for logging (use the original loss_mask before shifting)
        # We need to use the completion mask for mean logits
        chosen_mask = loss_mask[:num_examples, :-1]
        rejected_mask = loss_mask[num_examples:, :-1]
        
        if chosen_mask.any():
            mean_chosen_logits = logits[:num_examples, :-1][chosen_mask].mean()
        else:
            mean_chosen_logits = torch.tensor(0.0, device=logits.device)
        
        if rejected_mask.any():
            mean_rejected_logits = logits[num_examples:, :-1][rejected_mask].mean()
        else:
            mean_rejected_logits = torch.tensor(0.0, device=logits.device)
        
        return {
            "chosen_logps": chosen_logps,
            "rejected_logps": rejected_logps,
            "mean_chosen_logits": mean_chosen_logits,
            "mean_rejected_logits": mean_rejected_logits,
        }
    
    def compute_ref_log_probs(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute log probabilities from the reference model."""
        with torch.no_grad():
            if self.ref_model is None:
                ref_output = self.concatenated_forward(self.model, batch)
            else:
                ref_output = self.concatenated_forward(self.ref_model, batch)
        
        return ref_output["chosen_logps"], ref_output["rejected_logps"]
    
    def dpo_loss(
        self,
        chosen_logps: torch.Tensor,
        rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
        loss_type: str = "sigmoid",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute DPO loss.
        
        Args:
            chosen_logps: Log probs of chosen responses from policy
            rejected_logps: Log probs of rejected responses from policy
            ref_chosen_logps: Log probs of chosen responses from reference
            ref_rejected_logps: Log probs of rejected responses from reference
            loss_type: Type of loss function
            
        Returns:
            Tuple of (losses, chosen_rewards, rejected_rewards)
        """
        device = chosen_logps.device
        
        # Compute log ratios
        if self.reference_free:
            chosen_logratios = chosen_logps
            rejected_logratios = rejected_logps
        else:
            chosen_logratios = chosen_logps - ref_chosen_logps.to(device)
            rejected_logratios = rejected_logps - ref_rejected_logps.to(device)
        
        logits = chosen_logratios - rejected_logratios
        
        # Compute loss based on type
        if loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif loss_type == "ipo":
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif loss_type == "robust":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                + F.logsigmoid(-self.beta * logits) * self.label_smoothing
            ) / (1 - 2 * self.label_smoothing)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Compute rewards for logging
        chosen_rewards = self.beta * chosen_logratios.detach()
        rejected_rewards = self.beta * rejected_logratios.detach()
        
        return losses, chosen_rewards, rejected_rewards
    
    def get_batch_loss_metrics(
        self,
        model: VisionLanguageModel,
        batch: dict[str, Any],
        train_eval: Literal["train", "eval"] = "train",
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute DPO loss and metrics for a batch."""
        metrics = {}
        
        # Forward pass
        model_output = self.concatenated_forward(model, batch)
        
        # Get reference log probs
        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen_logps = batch["ref_chosen_logps"]
            ref_rejected_logps = batch["ref_rejected_logps"]
        else:
            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)
        
        # Compute loss
        losses = 0
        chosen_rewards = 0
        rejected_rewards = 0
        
        for loss_type in self.loss_type:
            _losses, _chosen_rewards, _rejected_rewards = self.dpo_loss(
                model_output["chosen_logps"],
                model_output["rejected_logps"],
                ref_chosen_logps,
                ref_rejected_logps,
                loss_type,
            )
            losses = losses + _losses
            chosen_rewards = chosen_rewards + _chosen_rewards
            rejected_rewards = rejected_rewards + _rejected_rewards
        
        # Compute accuracy
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        
        # Log metrics
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().item()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().item()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().item()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().item()
        metrics[f"{prefix}logps/chosen"] = model_output["chosen_logps"].detach().mean().item()
        metrics[f"{prefix}logps/rejected"] = model_output["rejected_logps"].detach().mean().item()
        metrics[f"{prefix}logits/chosen"] = model_output["mean_chosen_logits"].detach().item()
        metrics[f"{prefix}logits/rejected"] = model_output["mean_rejected_logits"].detach().item()
        
        return losses.mean(), metrics
    
    def compute_loss(
        self,
        model: VisionLanguageModel,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, float]]]:
        """Compute the DPO loss."""
        loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")
        
        # Store metrics
        self.store_metrics(metrics, train_eval="train")
        
        if return_outputs:
            return loss, metrics
        return loss
    
    def prediction_step(
        self,
        model: VisionLanguageModel,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Evaluation step."""
        with torch.no_grad():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")
        
        self.store_metrics(metrics, train_eval="eval")
        
        if prediction_loss_only:
            return loss.detach(), None, None
        
        # Return logits for evaluation
        logits = torch.tensor([
            metrics["eval_logits/chosen"],
            metrics["eval_logits/rejected"]
        ], device=self.accelerator.device)
        labels = torch.zeros(2, device=self.accelerator.device)
        
        return loss.detach(), logits, labels
    
    def store_metrics(
        self,
        metrics: dict[str, float],
        train_eval: Literal["train", "eval"] = "train",
    ) -> None:
        """Store metrics for logging."""
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)
    
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """Log metrics including DPO-specific ones."""
        train_eval = "train" if "loss" in logs else "eval"
        
        # Add stored metrics to logs so they appear in the table
        metrics_to_add = dict(self._stored_metrics[train_eval])
        for key, values in metrics_to_add.items():
            if values:
                logs[key] = sum(values) / len(values)
        
        # Clear stored metrics
        self._stored_metrics[train_eval].clear()
        
        return super().log(logs, start_time)
    
    def _set_signature_columns_if_needed(self):
        """Set signature columns for data collator."""
        if self._signature_columns is None:
            self._signature_columns = [
                "prompt_input_ids",
                "chosen_input_ids",
                "rejected_input_ids",
                "images",
                "image_ratios",
                "ref_chosen_logps",
                "ref_rejected_logps",
            ]
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Save the nanoVLM model using its native save_pretrained method.
        Handles both full fine-tuning and LoRA adapters.
        
        Args:
            output_dir: Directory to save the model. If None, uses args.output_dir.
            _internal_call: Whether this is an internal call from the Trainer.
        """
        if output_dir is None:
            output_dir = self.args.output_dir
        
        # Unwrap the model
        if hasattr(self, 'accelerator'):
            unwrapped_model = self.accelerator.unwrap_model(self.model)
        else:
            unwrapped_model = self.model
        
        # Check if model has LoRA adapters (PEFT)
        if hasattr(unwrapped_model, 'save_pretrained') and hasattr(unwrapped_model, 'peft_config'):
            # Model has PEFT LoRA adapters - save them
            print(f"Saving LoRA adapters to {output_dir}")
            unwrapped_model.save_pretrained(output_dir)
        else:
            # Use nanoVLM's native save method for full fine-tuning
            print(f"Saving full model to {output_dir}")
            unwrapped_model.save_pretrained(output_dir)
        
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)


class NanoVLMPreferenceDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for nanoVLM DPO training.
    
    Processes samples on-the-fly to avoid serialization issues with datasets.map().
    This is the recommended approach for nanoVLM DPO training.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        image_processor: Callable,
        tokenizer: PreTrainedTokenizerBase,
        model_cfg,
        max_prompt_length: Optional[int] = None,
        max_completion_length: Optional[int] = None,
    ):
        """
        Args:
            dataset: HuggingFace dataset with columns: (image or images), prompt, chosen, rejected
            image_processor: nanoVLM image processor from get_image_processor()
            tokenizer: nanoVLM tokenizer from get_tokenizer()
            model_cfg: nanoVLM model config (needs mp_image_token_length)
            max_prompt_length: Maximum prompt token length (optional)
            max_completion_length: Maximum completion token length (optional)
        
        Note:
            Supports both single and multiple images per sample:
            - Single image: "image" column with PIL Image
            - Multiple images: "images" column with list of PIL Images
        """
        self.dataset = dataset
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.mp_image_token_length = getattr(model_cfg, 'mp_image_token_length', 64)
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        from PIL import Image as PILImage
        
        item = self.dataset[idx]
        
        # Handle both single image and multiple images
        images_raw = item.get("images") or item.get("image")
        if images_raw is None:
            images_raw = []
        elif not isinstance(images_raw, list):
            images_raw = [images_raw]
        
        prompt = item["prompt"]
        chosen = item.get("chosen") or item.get("chosen_answer") or item.get("winner_answer")
        rejected = item.get("rejected") or item.get("rejected_answer") or item.get("loser_answer")
        
        # Load and process all images
        processed_images = []
        image_ratios = []
        
        for image in images_raw:
            # Load image if needed
            if isinstance(image, str):
                image = PILImage.open(image).convert("RGB")
            elif not isinstance(image, PILImage.Image):
                image = PILImage.fromarray(image).convert("RGB") if hasattr(image, '__array__') else image
            if hasattr(image, 'mode') and image.mode != "RGB":
                image = image.convert("RGB")
            
            # Process image
            processed_image, image_ratio = self.image_processor(image)
            
            # Handle global image token
            if not hasattr(self.tokenizer, "global_image_token"):
                if image_ratio[0] * image_ratio[1] == len(processed_image) - 1:
                    processed_image = processed_image[1:]
            
            processed_images.append(processed_image)
            image_ratios.append(image_ratio)
        
        # Get image string for all images
        image_string = get_image_string(self.tokenizer, image_ratios, self.mp_image_token_length) if image_ratios else ""
        
        # Create prompt with image tokens
        messages = [{"role": "user", "content": image_string + prompt}]
        
        # Tokenize prompt
        prompt_input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        
        # Tokenize completions
        chosen_input_ids = self.tokenizer(chosen, add_special_tokens=False)["input_ids"]
        rejected_input_ids = self.tokenizer(rejected, add_special_tokens=False)["input_ids"]
        
        # Add EOS token
        chosen_input_ids = chosen_input_ids + [self.tokenizer.eos_token_id]
        rejected_input_ids = rejected_input_ids + [self.tokenizer.eos_token_id]
        
        # Truncate if needed - but NEVER truncate image tokens!
        if self.max_prompt_length is not None and len(prompt_input_ids) > self.max_prompt_length:
            image_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.image_token)
            prompt_tensor = torch.tensor(prompt_input_ids)
            num_image_tokens = (prompt_tensor == image_token_id).sum().item()
            
            min_length_needed = num_image_tokens + 50
            if self.max_prompt_length >= min_length_needed:
                print(f"[WARNING] Prompt has {len(prompt_input_ids)} tokens with {num_image_tokens} image tokens. "
                      f"max_prompt_length={self.max_prompt_length} may truncate image tokens. Skipping truncation.")
            
        if self.max_completion_length is not None:
            chosen_input_ids = chosen_input_ids[:self.max_completion_length]
            rejected_input_ids = rejected_input_ids[:self.max_completion_length]
        
        # Stack all processed images or return None if no images
        if processed_images:
            stacked_images = torch.cat(processed_images, dim=0) if len(processed_images) > 1 else processed_images[0]
        else:
            stacked_images = None
        
        return {
            "prompt_input_ids": prompt_input_ids,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
            "images": stacked_images,
            "image_ratios": image_ratios,
        }


def create_nanovlm_dpo_dataset(
    dataset: Dataset,
    image_processor: Callable,
    tokenizer: PreTrainedTokenizerBase,
    model_cfg,
    max_prompt_length: Optional[int] = None,
    max_completion_length: Optional[int] = None,
) -> NanoVLMPreferenceDataset:
    """
    Create a PyTorch Dataset for nanoVLM DPO training.
    
    This wraps a HuggingFace dataset with on-the-fly processing,
    avoiding all serialization/multiprocessing issues.
    
    Args:
        dataset: HuggingFace dataset with columns: image, prompt, chosen, rejected
        image_processor: nanoVLM image processor from get_image_processor()
        tokenizer: nanoVLM tokenizer from get_tokenizer()
        model_cfg: nanoVLM model config
        max_prompt_length: Maximum prompt token length (optional)
        max_completion_length: Maximum completion token length (optional)
        
    Returns:
        NanoVLMPreferenceDataset ready for use with DataLoader or NanoVLMDPOTrainer
        
    Example:
        ```python
        from nanovlm.processors import get_tokenizer, get_image_processor
        
        tokenizer = get_tokenizer(model.cfg.lm_tokenizer, ...)
        image_processor = get_image_processor(model.cfg.max_img_size, ...)
        
        train_dataset = create_nanovlm_dpo_dataset(
            dataset=hf_dataset,
            image_processor=image_processor,
            tokenizer=tokenizer,
            model_cfg=model.cfg,
            max_prompt_length=256,
            max_completion_length=1024,
        )
        
        # Works immediately - no preprocessing needed
        print(train_dataset[0])  # Get first sample
        ```
    """
    return NanoVLMPreferenceDataset(
        dataset=dataset,
        image_processor=image_processor,
        tokenizer=tokenizer,
        model_cfg=model_cfg,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
    )


# =============================================================================
# Example Usage
# =============================================================================
"""
Example usage of NanoVLMDPOTrainer:

```python
import torch
from PIL import Image
from datasets import Dataset
from transformers import TrainingArguments

from nanovlm.vision_language_model import VisionLanguageModel
from nanovlm.processors import get_tokenizer, get_image_processor
from nanovlm_dpo_trainer import (
    NanoVLMDPOTrainer,
    NanoVLMDataCollatorForPreference,
    create_nanovlm_dpo_dataset,
    process_preference_sample,
)

# 1. Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionLanguageModel.from_pretrained("lusxvr/nanoVLM-230M-8k").to(device)

# 2. Get tokenizer and image processor
tokenizer = get_tokenizer(
    model.cfg.lm_tokenizer,
    model.cfg.vlm_extra_tokens,
    model.cfg.lm_chat_template
)
resize_to_max_side_len = getattr(model.cfg, "resize_to_max_side_len", False)
image_processor = get_image_processor(
    model.cfg.max_img_size,
    model.cfg.vit_img_size,
    resize_to_max_side_len
)

# 3. Prepare your preference dataset
# Your dataset should have columns: image, prompt, chosen, rejected
# Example:
raw_data = [
    {
        "image": Image.open("path/to/image1.jpg").convert("RGB"),
        "prompt": "What is shown in this image?",
        "chosen": "This image shows a beautiful sunset over the ocean.",
        "rejected": "I don't know what this image shows.",
    },
    # ... more samples
]
raw_dataset = Dataset.from_list(raw_data)

# 4. Process the dataset
processed_dataset = create_nanovlm_dpo_dataset(
    dataset=raw_dataset,
    tokenizer=tokenizer,
    image_processor=image_processor,
    model_cfg=model.cfg,
    max_prompt_length=512,
    max_completion_length=256,
    num_proc=1,  # Set to 1 for debugging, higher for production
)

# 5. Create training arguments
training_args = TrainingArguments(
    output_dir="./nanovlm-dpo-output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=100,
    remove_unused_columns=False,  # Important for custom columns
    bf16=True,  # Use bf16 if available
)

# 6. Create the trainer
trainer = NanoVLMDPOTrainer(
    model=model,
    ref_model=None,  # Will create a copy of the model
    args=training_args,
    train_dataset=processed_dataset,
    tokenizer=tokenizer,
    image_processor=image_processor,
    beta=0.1,
    loss_type="sigmoid",
    max_length=1024,
)

# 7. Train!
trainer.train()

# 8. Save the model
model.save_pretrained("./nanovlm-dpo-finetuned")
```

For manual processing of individual samples:

```python
# Process a single sample
sample = {
    "image": Image.open("path/to/image.jpg").convert("RGB"),
    "prompt": "Describe this image.",
    "chosen": "A detailed and accurate description.",
    "rejected": "An incorrect or vague description.",
}

processed = process_preference_sample(
    sample=sample,
    tokenizer=tokenizer,
    image_processor=image_processor,
    model_cfg=model.cfg,
    max_prompt_length=512,
    max_completion_length=256,
)

print(f"Prompt tokens: {len(processed['prompt_input_ids'])}")
print(f"Chosen tokens: {len(processed['chosen_input_ids'])}")
print(f"Rejected tokens: {len(processed['rejected_input_ids'])}")
print(f"Image shape: {processed['images'].shape}")
```
"""
