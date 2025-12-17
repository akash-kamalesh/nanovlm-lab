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
NanoVLM-compatible GRPO Trainer

This module adapts the HuggingFace TRL GRPOTrainer to work with nanoVLM models.
Adapted by Akash Kamalesh.

Key differences from the standard GRPOTrainer:
1. Uses nanoVLM's forward signature: forward(input_ids, images, attention_mask, targets)
2. Custom data collator for nanoVLM's image format
3. Uses nanoVLM's image processor and tokenizer
4. Handles nanoVLM's logits output format
5. Custom generation using nanoVLM's generate method
"""

import copy
import random
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import tqdm, gather, gather_object, set_seed
from datasets import Dataset, IterableDataset
from torch.utils.data import DataLoader, Sampler
from transformers import PreTrainedTokenizerBase, GenerationConfig
from transformers.data.data_collator import DataCollatorMixin
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput, seed_worker

# nanoVLM imports
from nanovlm.models.vision_language_model import VisionLanguageModel
from nanovlm.data.processors import get_tokenizer, get_image_processor, get_image_string
from nanovlm.models.config import VLMConfig


__all__ = [
    "NanoVLMGRPOTrainer",
    "NanoVLMGRPOConfig",
    "NanoVLMGRPODataCollator",
    "NanoVLMGRPODataset",
    "create_nanovlm_grpo_dataset",
    "pad",
    "selective_log_softmax",
    "entropy_from_logits",
]


# Type aliases
RewardFunc = Callable[[list, list], list[float]]


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


def selective_log_softmax(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute log softmax and gather log probabilities for labels."""
    log_probs = F.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy from logits."""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


@dataclass
class NanoVLMGRPOConfig:
    """Configuration for NanoVLM GRPO Trainer."""
    
    # Training parameters
    output_dir: str = "./nanovlm-grpo-output"
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-6
    num_train_epochs: int = 1
    max_steps: int = -1
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    weight_decay: float = 0.0
    logging_steps: int = 10
    save_steps: int = 100
    save_strategy: str = "steps"  # "no", "steps", "epoch"
    eval_steps: int = 100
    eval_strategy: str = "no"  # "no", "steps", "epoch"
    seed: int = 42
    bf16: bool = True
    fp16: bool = False
    
    # GRPO-specific parameters
    beta: float = 0.1  # KL penalty coefficient
    num_generations: int = 4  # Number of completions per prompt (G in GRPO paper)
    max_prompt_length: int = 512
    max_completion_length: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    
    # Loss configuration
    loss_type: str = "grpo"  # "grpo", "bnpo", "dr_grpo", "dapo", "cispo", "sapo"
    epsilon: float = 0.2  # Clipping parameter (epsilon_low)
    epsilon_high: Optional[float] = None  # Upper clipping bound (defaults to epsilon)
    delta: Optional[float] = None  # Optional upper bound for importance ratio (two-sided clipping)
    scale_rewards: str = "group"  # "group", "batch", "none"
    
    # SAPO-specific parameters
    sapo_temperature_pos: Optional[float] = None  # Temperature for positive advantages in SAPO
    sapo_temperature_neg: Optional[float] = None  # Temperature for negative advantages in SAPO
    
    # GSPO-specific parameters (enables GSPO when: importance_sampling_level="sequence", mask_truncated_completions=False, loss_type="dr_grpo")
    importance_sampling_level: str = "token"  # "token" or "sequence" - how to compute importance weights
    mask_truncated_completions: bool = True  # If True, mask out completions that were truncated (didn't end with EOS)
    
    # Reference model
    disable_dropout: bool = True
    
    # Logging
    log_completions: bool = True
    num_completions_to_print: int = 5
    
    # Multi-step
    num_iterations: int = 1  # μ in GRPO paper
    steps_per_generation: int = 1
    
    def __post_init__(self):
        if self.epsilon_high is None:
            self.epsilon_high = self.epsilon
        if self.loss_type == "sapo" and (self.sapo_temperature_neg is None or self.sapo_temperature_pos is None):
            raise ValueError(
                "When using `sapo` loss, both `sapo_temperature_neg` and `sapo_temperature_pos` must be set."
            )


@dataclass
class NanoVLMGRPODataCollator(DataCollatorMixin):
    """
    Data collator for nanoVLM GRPO data.
    
    Simply passes through the batch as GRPO handles its own batching.
    """
    tokenizer: PreTrainedTokenizerBase
    return_tensors: str = "pt"
    
    def torch_call(self, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # GRPO uses identity collator - just return the list
        return examples


class NanoVLMGRPOTrainer(Trainer):
    """
    GRPO Trainer adapted for nanoVLM models.
    
    This trainer implements Group Relative Policy Optimization (GRPO) for vision-language
    models using the nanoVLM architecture.
    
    Args:
        model: The nanoVLM model to train
        ref_model: Reference model for KL penalty (if None, a copy of model is created)
        reward_funcs: Reward function(s) for scoring completions
        args: GRPO configuration
        train_dataset: Training dataset with prompts
        eval_dataset: Evaluation dataset
        tokenizer: nanoVLM tokenizer
        image_processor: nanoVLM image processor
        data_collator: Data collator (default: NanoVLMGRPODataCollator)
        callbacks: Training callbacks
        optimizers: Optimizer and scheduler tuple
    """
    
    def __init__(
        self,
        model: VisionLanguageModel,
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        ref_model: Optional[VisionLanguageModel] = None,
        args: Optional[NanoVLMGRPOConfig] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        image_processor: Optional[Callable] = None,
        data_collator: Optional[DataCollatorMixin] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple = (None, None),
    ):
        # Default config
        if args is None:
            args = NanoVLMGRPOConfig()
        
        # Store GRPO config
        self.grpo_config = args
        
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
        
        # Get pad token id
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        
        # Create data collator if not provided
        if data_collator is None:
            data_collator = NanoVLMGRPODataCollator(tokenizer=tokenizer)
        
        # GRPO parameters
        self.beta = args.beta
        self.num_generations = args.num_generations
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.loss_type = args.loss_type
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high
        self.scale_rewards = args.scale_rewards
        self.num_iterations = args.num_iterations
        self.steps_per_generation = args.steps_per_generation
        self.log_completions = args.log_completions
        self.num_completions_to_print = args.num_completions_to_print
        self.importance_sampling_level = args.importance_sampling_level
        self.mask_truncated_completions = args.mask_truncated_completions
        self.delta = args.delta
        self.sapo_temperature_pos = args.sapo_temperature_pos
        self.sapo_temperature_neg = args.sapo_temperature_neg
        
        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_funcs = reward_funcs
        self.reward_func_names = [func.__name__ for func in reward_funcs]
        self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)
        
        # Reference model
        if ref_model is None and args.beta != 0.0:
            # Create a copy of the model as reference
            self.ref_model = copy.deepcopy(model)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
        else:
            self.ref_model = ref_model
        
        # Disable dropout if requested
        if args.disable_dropout:
            self._disable_dropout(model)
            if self.ref_model is not None:
                self._disable_dropout(self.ref_model)
        
        # Tracking - use _stored_metrics pattern like DPO trainer for proper logging
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self._step = 0
        self._buffered_inputs = None
        self._logs = {
            "prompt": [],
            "completion": [],
            "rewards": defaultdict(list),
            "advantages": [],
            "images": [],
        }
        
        # Create training arguments for parent Trainer
        from transformers import TrainingArguments
        
        # Determine eval strategy
        eval_strategy = args.eval_strategy if eval_dataset is not None else "no"
        eval_steps_value = args.eval_steps if eval_dataset is not None and eval_strategy != "no" else None
        
        # Determine save strategy
        save_strategy = args.save_strategy
        save_steps_value = args.save_steps if save_strategy == "steps" else None
        
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            logging_steps=args.logging_steps,
            save_steps=save_steps_value,
            save_strategy=save_strategy,
            eval_steps=eval_steps_value,
            eval_strategy=eval_strategy,
            seed=args.seed,
            bf16=args.bf16,
            fp16=args.fp16,
            remove_unused_columns=False,
            report_to="none",  # Handle logging ourselves
        )
        
        # Set seed
        set_seed(args.seed)
        
        # Initialize parent trainer
        super().__init__(
            model=model,
            args=training_args,
            data_collator=lambda x: x,  # Identity collator for GRPO
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        
        # Move ref model to device after accelerator setup
        if self.ref_model is not None:
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        
        # Generation config
        self.generation_config = {
            "max_new_tokens": self.max_completion_length,
            "do_sample": True,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k if self.top_k > 0 else None,
        }
    
    def _disable_dropout(self, model: nn.Module):
        """Disable dropout in the model."""
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
    
    def _set_signature_columns_if_needed(self):
        """Set signature columns for data collator."""
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "image", "images"]
    
    def get_train_dataloader(self) -> DataLoader:
        """Create training dataloader with GRPO-specific batching."""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_dataset = self.train_dataset
        
        # Batch size includes steps_per_generation
        batch_size = self.args.per_device_train_batch_size * self.steps_per_generation
        
        dataloader_params = {
            "batch_size": batch_size,
            "collate_fn": lambda x: x,  # Identity collator
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "shuffle": True,
        }
        
        dataloader = DataLoader(train_dataset, **dataloader_params)
        return self.accelerator.prepare(dataloader)
    
    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        """Create evaluation dataloader."""
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        
        # For evaluation, use per_device_eval_batch_size
        batch_size = self.args.per_device_eval_batch_size
        
        dataloader_params = {
            "batch_size": batch_size,
            "collate_fn": lambda x: x,  # Identity collator
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "shuffle": False,
        }
        
        dataloader = DataLoader(eval_dataset, **dataloader_params)
        return self.accelerator.prepare(dataloader)
    
    def _process_image(self, image) -> tuple[torch.Tensor, tuple]:
        """Process a single image using nanoVLM's image processor."""
        from PIL import Image as PILImage
        
        # Load image if needed
        if isinstance(image, str):
            image = PILImage.open(image).convert("RGB")
        elif not isinstance(image, PILImage.Image):
            if hasattr(image, '__array__'):
                image = PILImage.fromarray(np.array(image)).convert("RGB")
        if hasattr(image, 'mode') and image.mode != "RGB":
            image = image.convert("RGB")
        
        # Process image
        processed_image, image_ratio = self.image_processor(image)
        
        # Handle global image token
        if not hasattr(self.processing_class, "global_image_token"):
            if image_ratio[0] * image_ratio[1] == len(processed_image) - 1:
                processed_image = processed_image[1:]
        
        return processed_image, image_ratio
    
    def _prepare_prompt(self, prompt: str, image_ratios: list[tuple]) -> list[int]:
        """Prepare prompt with image tokens for nanoVLM.
        
        Args:
            prompt: The text prompt
            image_ratios: List of (n_h, n_w) tuples for each image
        """
        mp_image_token_length = getattr(self.model_cfg, 'mp_image_token_length', 64)
        
        # Get image string for prompt - image_ratios is already a list of tuples
        image_string = get_image_string(self.processing_class, image_ratios, mp_image_token_length)
        
        # Create prompt with image tokens
        messages = [{"role": "user", "content": image_string + prompt}]
        
        # Tokenize prompt
        prompt_input_ids = self.processing_class.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        
        # Truncate if needed
        if self.max_prompt_length is not None and len(prompt_input_ids) > self.max_prompt_length:
            # Count image tokens
            image_token_id = self.processing_class.convert_tokens_to_ids(self.processing_class.image_token)
            num_image_tokens = sum(1 for t in prompt_input_ids if t == image_token_id)
            
            # Only truncate if we can preserve image tokens
            if self.max_prompt_length >= num_image_tokens + 20:
                # Truncate from the middle (keep start and end)
                excess = len(prompt_input_ids) - self.max_prompt_length
                mid = len(prompt_input_ids) // 2
                prompt_input_ids = prompt_input_ids[:mid - excess//2] + prompt_input_ids[mid + excess//2 + excess%2:]
        
        return prompt_input_ids
    
    @torch.no_grad()
    def _generate_completions(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: list[torch.Tensor],
        num_generations: int,
    ) -> tuple[list[list[int]], list[list[float]]]:
        """
        Generate completions for prompts using nanoVLM.
        
        Returns:
            completion_ids: List of completion token IDs
            logprobs: List of per-token log probabilities (or None)
        """
        device = self.accelerator.device
        
        # Unwrap model
        if hasattr(self, 'accelerator'):
            unwrapped_model = self.accelerator.unwrap_model(self.model)
        else:
            unwrapped_model = self.model
        
        all_completion_ids = []
        
        batch_size = prompt_ids.size(0)
        
        # nanoVLM's attention doesn't support both is_causal=True and attn_mask
        # Only pass attention_mask if there's actual padding (not all 1s)
        if attention_mask is not None and attention_mask.all():
            attention_mask_to_use = None
        else:
            attention_mask_to_use = attention_mask
        
        # Move images to device if needed
        if images is not None:
            # Filter out None values but keep track of which positions have images
            images_on_device = []
            for img in images:
                if img is not None:
                    if isinstance(img, torch.Tensor):
                        images_on_device.append(img.to(device))
                    else:
                        images_on_device.append(img)
            
            # Stack images into a single tensor for nanoVLM
            # nanoVLM expects images as tensor [num_images, C, H, W] or list
            if len(images_on_device) > 0:
                images_tensor = torch.cat([img.unsqueeze(0) if img.dim() == 3 else img for img in images_on_device], dim=0)
            else:
                images_tensor = None
        else:
            images_tensor = None
        
        # Single batched generation call (prompts are already repeated for num_generations)
        # Note: num_generations param here is typically 1 since batching is done upstream
        generated_ids = unwrapped_model.generate(
            prompt_ids,  # input_ids
            images_tensor,  # images tensor
            max_new_tokens=self.max_completion_length,
            temperature=self.temperature if self.temperature > 0 else 1.0,
            top_k=self.top_k if self.top_k > 0 else 50,
            top_p=self.top_p if self.top_p < 1.0 else 0.95,
        )
        
        # nanoVLM's generate returns ONLY the new tokens, not the full sequence
        for i in range(batch_size):
            completion = generated_ids[i].tolist()
            
            # Truncate at EOS if present
            if self.eos_token_id in completion:
                eos_idx = completion.index(self.eos_token_id)
                completion = completion[:eos_idx + 1]
            
            # Ensure we have at least one token
            if len(completion) == 0:
                completion = [self.eos_token_id]
            
            all_completion_ids.append(completion)
        
        return all_completion_ids, None  # logprobs not computed during generation
    
    def _get_per_token_logps(
        self,
        model: VisionLanguageModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: list[torch.Tensor],
        logits_to_keep: int,
    ) -> torch.Tensor:
        """Compute per-token log probabilities for completions."""
        device = input_ids.device
        
        # Unwrap model
        if hasattr(self, 'accelerator'):
            unwrapped_model = self.accelerator.unwrap_model(model)
        else:
            unwrapped_model = model
        
        # Prepare images tensor for nanoVLM
        if images is not None and len(images) > 0:
            # Filter out None values
            images_on_device = []
            for img in images:
                if img is not None:
                    if isinstance(img, torch.Tensor):
                        images_on_device.append(img.to(device))
                    else:
                        images_on_device.append(img)
            
            if len(images_on_device) > 0:
                images_tensor = torch.cat([img.unsqueeze(0) if img.dim() == 3 else img for img in images_on_device], dim=0)
            else:
                images_tensor = None
        else:
            images_tensor = None
        
        # Forward pass through nanoVLM - ALWAYS pass attention_mask=None
        # nanoVLM uses is_causal=True internally and conflicts with explicit masks
        hidden_states, _ = unwrapped_model(
            input_ids=input_ids,
            images=images_tensor,
            attention_mask=None,
            targets=None,
        )
        
        # Apply LM head to get logits
        logits = unwrapped_model.decoder.head(hidden_states)
        
        # For causal LM, logits at position i predict token at position i+1
        seq_len = logits.size(1)
        
        # Handle edge case
        if logits_to_keep == 0 or seq_len < 2:
            return torch.zeros(input_ids.size(0), 0, device=input_ids.device)
        
        # Slice logits to get predictions for completion tokens
        start_idx = max(0, seq_len - logits_to_keep - 1)
        end_idx = seq_len - 1
        logits = logits[:, start_idx:end_idx, :]
        
        # Divide by temperature
        logits = logits / self.temperature
        
        # Get completion token IDs
        completion_ids = input_ids[:, -logits_to_keep:]
        
        # Align sizes if needed
        actual_logits_len = logits.size(1)
        if actual_logits_len < completion_ids.size(1):
            completion_ids = completion_ids[:, -actual_logits_len:]
        
        # Compute log probabilities
        per_token_logps = selective_log_softmax(logits, completion_ids)
        
        return per_token_logps
    
    def _get_per_token_logps_and_entropies(
        self,
        model: VisionLanguageModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: list[torch.Tensor],
        logits_to_keep: int,
        compute_entropy: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute per-token log probabilities and optionally entropies."""
        device = input_ids.device
        
        # Unwrap model
        if hasattr(self, 'accelerator'):
            unwrapped_model = self.accelerator.unwrap_model(model)
        else:
            unwrapped_model = model
        
        # Prepare images tensor for nanoVLM
        if images is not None and len(images) > 0:
            # Filter out None values
            images_on_device = []
            for img in images:
                if img is not None:
                    if isinstance(img, torch.Tensor):
                        images_on_device.append(img.to(device))
                    else:
                        images_on_device.append(img)
            
            if len(images_on_device) > 0:
                images_tensor = torch.cat([img.unsqueeze(0) if img.dim() == 3 else img for img in images_on_device], dim=0)
            else:
                images_tensor = None
        else:
            images_tensor = None
        
        # Forward pass through nanoVLM - ALWAYS pass attention_mask=None
        # nanoVLM uses is_causal=True internally and conflicts with explicit masks
        hidden_states, _ = unwrapped_model(
            input_ids=input_ids,
            images=images_tensor,
            attention_mask=None,
            targets=None,
        )
        
        # Apply LM head to get logits
        logits = unwrapped_model.decoder.head(hidden_states)
        
        # For causal LM, logits at position i predict token at position i+1
        # We want logits that predict the completion tokens
        # If input_ids = [prompt..., completion...], completion starts at position (seq_len - logits_to_keep)
        # So we need logits from position (seq_len - logits_to_keep - 1) to (seq_len - 2)
        seq_len = logits.size(1)
        
        # Handle edge case where logits_to_keep might be 0 or larger than available
        if logits_to_keep == 0 or seq_len < 2:
            per_token_logps = torch.zeros(input_ids.size(0), 0, device=input_ids.device)
            entropies = torch.zeros_like(per_token_logps) if compute_entropy else None
            return per_token_logps, entropies
        
        # Slice logits: we need positions that predict completion tokens
        # completion tokens are at positions [seq_len - logits_to_keep, seq_len - 1]
        # so we need logits at positions [seq_len - logits_to_keep - 1, seq_len - 2]
        start_idx = max(0, seq_len - logits_to_keep - 1)
        end_idx = seq_len - 1
        logits = logits[:, start_idx:end_idx, :]
        
        # Divide by temperature
        logits = logits / self.temperature
        
        # Get completion token IDs (the tokens we're predicting)
        # These are at positions [seq_len - logits_to_keep, seq_len - 1] in input_ids
        completion_ids = input_ids[:, -logits_to_keep:]
        
        # Align sizes if needed (in case start_idx was clamped)
        actual_logits_len = logits.size(1)
        if actual_logits_len < completion_ids.size(1):
            completion_ids = completion_ids[:, -actual_logits_len:]
        
        # Compute log probabilities
        per_token_logps = selective_log_softmax(logits, completion_ids)
        
        # Compute entropy if requested
        if compute_entropy:
            with torch.no_grad():
                entropies = entropy_from_logits(logits)
        else:
            entropies = None
        
        return per_token_logps, entropies
    
    def _calculate_rewards(
        self,
        inputs: list[dict],
        prompts: list[str],
        completions: list[str],
        completion_ids_list: list[list[int]],
    ) -> torch.Tensor:
        """Calculate rewards for completions using reward functions."""
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        
        # Prepare kwargs for reward functions
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
        
        # Map common answer field names to 'answer' for reward functions
        if "original_answer" in reward_kwargs and "answer" not in reward_kwargs:
            reward_kwargs["answer"] = reward_kwargs.pop("original_answer")
        elif "expected_answer" in reward_kwargs and "answer" not in reward_kwargs:
            reward_kwargs["answer"] = reward_kwargs.pop("expected_answer")
        
        for i, reward_func in enumerate(self.reward_funcs):
            output_reward_func = reward_func(
                prompts=prompts,
                completions=completions,
                completion_ids=completion_ids_list,
                **reward_kwargs
            )
            # Convert None values to NaN
            output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
        
        # Gather rewards across processes
        rewards_per_func = gather(rewards_per_func)
        return rewards_per_func
    
    def _generate_and_score_completions(
        self,
        inputs: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        """Generate completions and compute rewards/advantages."""
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        
        # Debug input format
        # print(f"[DEBUG] _generate_and_score_completions inputs type: {type(inputs)}")
        # if isinstance(inputs, list) and len(inputs) > 0:
        #     print(f"[DEBUG] First element type: {type(inputs[0])}, keys: {inputs[0].keys() if isinstance(inputs[0], dict) else 'N/A'}")
        
        # Handle different input formats
        if isinstance(inputs, dict):
            # Single sample, wrap in list
            inputs = [inputs]
        elif not isinstance(inputs, list):
            raise TypeError(f"Expected inputs to be list or dict, got {type(inputs)}")
        
        # Ensure each element is a dict
        if len(inputs) > 0 and not isinstance(inputs[0], dict):
            raise TypeError(f"Expected inputs elements to be dicts, got {type(inputs[0])}")
        
        # Extract prompts and images
        prompts = [x["prompt"] for x in inputs]
        
        # Handle images - support both single and multiple images
        if "images" in inputs[0]:
            raw_images = [x.get("images") for x in inputs]
        elif "image" in inputs[0]:
            raw_images = [x.get("image") for x in inputs]
        else:
            raw_images = None
        
        # Process images and prepare prompts
        all_prompt_ids = []
        all_images = []
        all_image_ratios = []
        
        for i, prompt in enumerate(prompts):
            if raw_images is not None and raw_images[i] is not None:
                # Handle both single image and list of images
                images_to_process = raw_images[i] if isinstance(raw_images[i], list) else [raw_images[i]]
                
                # Process all images
                processed_images = []
                image_ratios = []
                for image in images_to_process:
                    processed_image, image_ratio = self._process_image(image)
                    processed_images.append(processed_image)
                    image_ratios.append(image_ratio)
                
                # Stack images if multiple
                if len(processed_images) > 1:
                    stacked_image = torch.cat(processed_images, dim=0)
                else:
                    stacked_image = processed_images[0]
                
                all_images.append(stacked_image)
                all_image_ratios.append(image_ratios)
                
                # Prepare prompt with image tokens for all images
                prompt_ids = self._prepare_prompt(prompt, image_ratios)
            else:
                all_images.append(None)
                all_image_ratios.append(None)
                # Tokenize prompt without image
                messages = [{"role": "user", "content": prompt}]
                prompt_ids = self.processing_class.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
            
            all_prompt_ids.append(prompt_ids)
        
        # Repeat prompts for num_generations
        repeated_prompt_ids = []
        repeated_images = []
        repeated_inputs = []
        
        for i in range(len(prompts)):
            for _ in range(self.num_generations):
                repeated_prompt_ids.append(all_prompt_ids[i])
                repeated_images.append(all_images[i])
                repeated_inputs.append(inputs[i].copy())
        
        # Pad prompt IDs
        prompt_ids_tensors = [torch.tensor(ids, dtype=torch.long, device=device) for ids in repeated_prompt_ids]
        prompt_mask_tensors = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids_tensors]
        prompt_ids = pad(prompt_ids_tensors, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask_tensors, padding_value=0, padding_side="left")
        
        # Prepare images for batch
        # Pass the full list with None values to maintain correspondence with prompt_ids
        batch_images = repeated_images if any(img is not None for img in repeated_images) else None
        
        # Generate completions
        completion_ids_list, _ = self._generate_completions(
            prompt_ids, prompt_mask, batch_images, num_generations=1  # Already repeated
        )
        
        # Pad completion IDs
        completion_ids_tensors = [torch.tensor(ids, dtype=torch.long, device=device) for ids in completion_ids_list]
        completion_mask_tensors = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids_tensors]
        completion_ids = pad(completion_ids_tensors, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask_tensors, padding_value=0, padding_side="right")
        
        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        # A completion is truncated if it doesn't end with EOS or pad token
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor(
                [ids[-1] not in eos_and_pad for ids in completion_ids_list], 
                device=device
            )
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()
        
        # Clear intermediate tensors early to save VRAM
        del completion_ids_tensors, completion_mask_tensors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Decode completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        prompts_text = [prompts[i // self.num_generations] for i in range(len(completion_ids_list))]
        
        # Calculate rewards
        rewards_per_func = self._calculate_rewards(
            repeated_inputs, prompts_text, completions_text, completion_ids_list
        )
        
        # Apply weights and sum rewards
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        
        # Compute grouped-wise rewards and advantages
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        
        # Scale rewards
        if self.scale_rewards == "group":
            std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
        elif self.scale_rewards == "batch":
            std_rewards = rewards.std().expand_as(rewards)
        else:
            std_rewards = torch.ones_like(rewards)
        
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)
        
        # Slice for local process
        process_slice = slice(
            self.accelerator.process_index * len(repeated_inputs),
            (self.accelerator.process_index + 1) * len(repeated_inputs),
        )
        advantages = advantages[process_slice]
        
        # Log metrics - Completion statistics
        completion_lengths = torch.tensor([len(ids) for ids in completion_ids_list], device=device)
        
        # Completion length statistics
        self._stored_metrics[mode]["len/mean"].append(completion_lengths.float().mean().item())
        self._stored_metrics[mode]["len/min"].append(completion_lengths.float().min().item())
        self._stored_metrics[mode]["len/max"].append(completion_lengths.float().max().item())
        
        # Clipped ratio (completions that were truncated)
        eos_and_pad = [self.eos_token_id, self.pad_token_id]
        is_truncated = torch.tensor(
            [ids[-1] not in eos_and_pad for ids in completion_ids_list], 
            device=device
        )
        clipped_ratio = is_truncated.float().mean().item()
        self._stored_metrics[mode]["clip_ratio"].append(clipped_ratio)
        
        # Terminated length statistics (for non-truncated completions)
        if (~is_truncated).any():
            terminated_lengths = completion_lengths[~is_truncated]
            self._stored_metrics[mode]["term_len/mean"].append(terminated_lengths.float().mean().item())
            self._stored_metrics[mode]["term_len/min"].append(terminated_lengths.float().min().item())
            self._stored_metrics[mode]["term_len/max"].append(terminated_lengths.float().max().item())
        else:
            # If all completions are truncated, use completion lengths
            self._stored_metrics[mode]["term_len/mean"].append(completion_lengths.float().mean().item())
            self._stored_metrics[mode]["term_len/min"].append(completion_lengths.float().min().item())
            self._stored_metrics[mode]["term_len/max"].append(completion_lengths.float().max().item())
        
        # Reward statistics - mean and std for overall reward
        reward_mean = mean_grouped_rewards.mean().item()
        reward_std = rewards.std().item()
        self._stored_metrics[mode]["reward"].append(reward_mean)
        self._stored_metrics[mode]["reward_std"].append(reward_std)
        
        # Per-reward-function statistics
        for i, reward_func_name in enumerate(self.reward_func_names):
            reward_values = rewards_per_func[:, i]
            # Filter out NaN values
            valid_rewards = reward_values[~torch.isnan(reward_values)]
            if len(valid_rewards) > 0:
                func_mean = valid_rewards.mean().item()
                func_std = valid_rewards.std().item()
            else:
                func_mean = 0.0
                func_std = 0.0
            self._stored_metrics[mode][f"rewards/{reward_func_name}/mean"].append(func_mean)
            self._stored_metrics[mode][f"rewards/{reward_func_name}/std"].append(func_std)
        
        # Log completions
        self._logs["prompt"].extend(prompts_text)
        self._logs["completion"].extend(completions_text)
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(advantages.tolist())
        
        # Concatenate prompt and completion
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        
        # Compute reference log probs if needed
        logits_to_keep = completion_ids.size(1)
        
        with torch.no_grad():
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        batch_images,
                        logits_to_keep,
                    )
                else:
                    ref_per_token_logps = None
            else:
                ref_per_token_logps = None
        # Compute old_per_token_logps only when needed (multi-iteration training)
        # When num_iterations == 1, we can use per_token_logps.detach() in compute_loss
        if self.num_iterations > 1:
            with torch.no_grad():
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model, prompt_completion_ids, attention_mask, batch_images, logits_to_keep
                )
        else:
            old_per_token_logps = None
        # Compute num_items_in_batch (total completion tokens) for DAPO/CISPO loss
        completion_lengths = torch.tensor([len(ids) for ids in completion_ids_list], device=device)
        agg_completion_lengths = gather(completion_lengths)
        num_items_in_batch = agg_completion_lengths.sum().item()
        
        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "images": batch_images,
            "num_items_in_batch": num_items_in_batch,
        }
        
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        
        return output
    
    def _prepare_inputs(self, inputs: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Prepare inputs by generating completions and computing advantages."""
        mode = "train" if self.model.training else "eval"
        
        if mode == "train":
            generate_every = self.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                generation_batch = self._generate_and_score_completions(inputs)
                self._buffered_inputs = [generation_batch]
            return self._buffered_inputs[self._step % self.steps_per_generation]
        else:
            return self._generate_and_score_completions(inputs)
    
    @staticmethod
    def _get_sapo_token_loss(unclipped_token_loss: torch.Tensor, temperature: float) -> torch.Tensor:
        """Compute SAPO token loss with sigmoid smoothing."""
        sigmoid_input = temperature * (unclipped_token_loss - 1)
        sigmoid_smoothed_loss = torch.nn.functional.sigmoid(sigmoid_input)
        sapo_token_loss = sigmoid_smoothed_loss * 4 / temperature
        return sapo_token_loss
    
    def compute_loss(
        self,
        model: VisionLanguageModel,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute GRPO loss."""
        # Get inputs
        prompt_ids = inputs["prompt_ids"]
        prompt_mask = inputs["prompt_mask"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        advantages = inputs["advantages"]
        images = inputs.get("images")
        
        # Concatenate prompt and completion
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        
        logits_to_keep = completion_ids.size(1)
        
        # Compute per-token log probs and entropies
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            images,
            logits_to_keep,
            compute_entropy=True,
        )
        
        # Compute KL divergence if beta != 0
        if self.beta != 0.0 and "ref_per_token_logps" in inputs:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) 
                - (ref_per_token_logps - per_token_logps) - 1
            )
        else:
            per_token_kl = None
        
        # Compute loss based on loss type
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)
        
        # Get old log probs for importance sampling
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps
        
        # Compute importance sampling weights based on importance_sampling_level
        log_ratio = per_token_logps - old_per_token_logps
        completion_mask_float = completion_mask.float()
        
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            # Sequence-level: average log ratio across tokens, then broadcast back
            log_importance_weights = (log_ratio * completion_mask_float).sum(-1) / completion_mask_float.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. "
                "Possible values are 'token' and 'sequence'."
            )
        
        coef_1 = torch.exp(log_importance_weights)
        
        # Compute per-token loss based on loss type
        if self.loss_type == "cispo":
            # CISPO uses clamped importance weights
            clamped_ratios = torch.clamp(coef_1, max=self.epsilon_high).detach()
            per_token_loss = -clamped_ratios * advantages * per_token_logps
        elif self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo"]:
            # Standard PPO-style clipping
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            # Two-sided clipping with delta (optional upper bound for importance ratio)
            if self.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.delta)
            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        elif self.loss_type == "sapo":
            # SAPO: Sigmoid-based Advantage Policy Optimization
            per_token_loss = torch.empty_like(coef_1)
            positive_advantages_mask = advantages.repeat([1, coef_1.shape[1]]) > 0
            per_token_loss[positive_advantages_mask] = self._get_sapo_token_loss(
                coef_1[positive_advantages_mask], self.sapo_temperature_pos
            )
            per_token_loss[~positive_advantages_mask] = self._get_sapo_token_loss(
                coef_1[~positive_advantages_mask], self.sapo_temperature_neg
            )
            per_token_loss = -per_token_loss * advantages
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Add KL penalty
        if self.beta != 0.0 and per_token_kl is not None:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        
        # Compute final loss based on loss type
        if self.loss_type in ["grpo", "sapo"]:
            # Per-sequence mean, then batch mean
            loss = ((per_token_loss * completion_mask_float).sum(-1) / completion_mask_float.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            # Global mean across all tokens
            loss = (per_token_loss * completion_mask_float).sum() / completion_mask_float.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            # DR-GRPO: normalize by batch_size * max_completion_length (GSPO uses this)
            loss = (per_token_loss * completion_mask_float).sum() / (per_token_loss.size(0) * self.max_completion_length)
        elif self.loss_type in ["cispo", "dapo"]:
            # DAPO/CISPO: normalize by num_items_in_batch
            normalizer = num_items_in_batch if num_items_in_batch is not None else per_token_loss.size(0)
            loss = (per_token_loss * completion_mask_float).sum() / normalizer
        else:
            # Fallback
            loss = ((per_token_loss * completion_mask_float).sum(-1) / completion_mask_float.sum(-1).clamp(min=1.0)).mean()
        
        # Scale by gradient accumulation (use current value which may differ for last batch)
        current_grad_accum = getattr(self, 'current_gradient_accumulation_steps', self.args.gradient_accumulation_steps)
        loss = loss / current_grad_accum
        
        # Log metrics
        mode = "train" if self.model.training else "eval"
        
        if self.beta != 0.0 and per_token_kl is not None:
            mean_kl = (per_token_kl * completion_mask_float).sum() / completion_mask_float.sum().clamp(min=1.0)
            self._stored_metrics[mode]["kl"].append(mean_kl.item())
        else:
            # Log 0 for KL if not computed
            self._stored_metrics[mode]["kl"].append(0.0)
        
        if entropies is not None:
            mean_entropy = (entropies * completion_mask_float).sum() / completion_mask_float.sum().clamp(min=1.0)
            self._stored_metrics[mode]["entropy"].append(mean_entropy.item())
        
        # Log clip ratios based on loss type
        completion_token_count = completion_mask_float.sum().clamp(min=1.0)
        
        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * completion_mask_float).sum() / completion_token_count
        
        if self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo"]:
            # Compute clipped probability ratios for GRPO-family loss types
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped
            
            low_clip = masked_batch_mean(is_low_clipped.float())
            high_clip = masked_batch_mean(is_high_clipped.float())
            clip_ratio = masked_batch_mean(is_region_clipped.float())
            
            self._stored_metrics[mode]["clip_ratio/low"].append(low_clip.item())
            self._stored_metrics[mode]["clip_ratio/high"].append(high_clip.item())
            self._stored_metrics[mode]["clip_ratio/region"].append(clip_ratio.item())
        elif self.loss_type == "cispo":
            # CISPO-specific clip ratio logging
            is_cispo_clipped = (coef_1 > self.epsilon_high) & (advantages > 0)
            cispo_clip_ratio = masked_batch_mean(is_cispo_clipped.float())
            self._stored_metrics[mode]["cispo_clip_ratio"].append(cispo_clip_ratio.item())
        
        return loss
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training step to track step count."""
        self._step += 1
        
        # Debug: print type of inputs (uncomment to debug)
        # if self._step == 1:
        #     print(f"[DEBUG] training_step inputs type: {type(inputs)}")
        #     if isinstance(inputs, list):
        #         print(f"[DEBUG] inputs length: {len(inputs)}, first element type: {type(inputs[0]) if inputs else 'empty'}")
        #     elif isinstance(inputs, dict):
        #         print(f"[DEBUG] inputs keys: {inputs.keys()}")
        
        # Handle case where inputs might be wrapped differently
        if isinstance(inputs, dict):
            # If it's already a dict with our expected keys, it's already prepared
            if "prompt_ids" in inputs:
                prepared_inputs = inputs
            else:
                # It's a single sample dict, wrap in list
                prepared_inputs = self._prepare_inputs([inputs])
        elif isinstance(inputs, list):
            prepared_inputs = self._prepare_inputs(inputs)
        else:
            # Unexpected format, try to handle
            raise TypeError(f"Unexpected inputs type: {type(inputs)}. Expected list of dicts or dict.")
        
        # Ensure model is in training mode
        model.train()
        
        # Compute loss with gradients enabled
        # Pass num_items_in_batch for DAPO/CISPO loss types
        num_items = prepared_inputs.get("num_items_in_batch")
        loss = self.compute_loss(model, prepared_inputs, num_items_in_batch=num_items)
        
        # Backward pass
        self.accelerator.backward(loss)
        
        # Clear intermediate tensors to save VRAM
        del prepared_inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return loss.detach()
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Evaluation step."""
        # Set model to eval mode
        model.eval()
        
        prepared_inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            num_items = prepared_inputs.get("num_items_in_batch")
            loss = self.compute_loss(model, prepared_inputs, num_items_in_batch=num_items)
        return loss.detach(), None, None
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        """Evaluate the model and return evaluation results with custom metrics."""
        # Ensure model is in eval mode
        self.model.eval()
        
        # Clear eval metrics before evaluation
        self._stored_metrics["eval"].clear()
        
        # Call parent evaluate which will use our prediction_step and log methods
        eval_loop_output = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        
        return eval_loop_output
    
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """Log metrics including GRPO-specific ones."""
        # Determine mode based on whether "loss" is in logs (training) or not (evaluation)
        train_eval = "train" if "loss" in logs else "eval"
        
        # Add stored metrics to logs so they appear in the table (DPO trainer pattern)
        metrics_to_add = dict(self._stored_metrics[train_eval])
        for key, values in metrics_to_add.items():
            if values:
                avg_value = sum(values) / len(values)
                
                # Handle metric naming
                log_key = key
                
                # Rename 'kl' to 'kl_divergence' to avoid blank header issues with short keys
                if log_key == "kl":
                    log_key = "kl_divergence"
                
                # Prefix with eval_ if in eval mode (standard HF Trainer behavior)
                if train_eval == "eval" and not log_key.startswith("eval_"):
                    log_key = f"eval_{log_key}"
                
                logs[log_key] = avg_value
        
        # Print sample completions (optional verbose logging) - only for training
        if self.log_completions and self._logs["prompt"] and train_eval == "train":
            print("\n" + "-"*60)
            print("SAMPLE COMPLETIONS")
            print("-"*60)
            for i in range(min(self.num_completions_to_print, len(self._logs["prompt"]))):
                print(f"\nPrompt: {self._logs['prompt'][i]}")
                print(f"Completion: {self._logs['completion'][i]}")
                for name in self.reward_func_names:
                    if name in self._logs["rewards"] and i < len(self._logs["rewards"][name]):
                        print(f"Reward ({name}): {self._logs['rewards'][name][i]:.4f}")
                if i < len(self._logs["advantages"]):
                    print(f"Advantage: {self._logs['advantages'][i]:.4f}")
            print("-"*60 + "\n")
        
        # Print sample completions for evaluation if log_completions is enabled
        if self.log_completions and self._logs["prompt"] and train_eval == "eval":
            print("\n" + "-"*60)
            print("EVALUATION SAMPLE COMPLETIONS")
            print("-"*60)
            for i in range(min(self.num_completions_to_print, len(self._logs["prompt"]))):
                print(f"\nPrompt: {self._logs['prompt'][i]}")
                print(f"Completion: {self._logs['completion'][i]}")
                for name in self.reward_func_names:
                    if name in self._logs["rewards"] and i < len(self._logs["rewards"][name]):
                        print(f"Reward ({name}): {self._logs['rewards'][name][i]:.4f}")
                if i < len(self._logs["advantages"]):
                    print(f"Advantage: {self._logs['advantages'][i]:.4f}")
            print("-"*60 + "\n")
        
        # Clear stored metrics
        self._stored_metrics[train_eval].clear()
        
        self._logs = {
            "prompt": [],
            "completion": [],
            "rewards": defaultdict(list),
            "advantages": [],
            "images": [],
        }
        
        return super().log(logs, start_time)
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Save the nanoVLM model with proper strategy handling."""
        if output_dir is None:
            output_dir = self.args.output_dir
        
        # Check if we should save based on save_strategy
        if self.args.save_strategy == "no":
            return
        
        # # Unwrap the model if needed
        # if hasattr(self, 'accelerator'):
        #     unwrapped_model = self.accelerator.unwrap_model(self.model)
        # else:
        #     unwrapped_model = self.model
        
        # Only save on main process
        if self.accelerator.is_main_process:
            self.model.save_pretrained(output_dir)
            
            # Save tokenizer
            if self.processing_class is not None:
                self.processing_class.save_pretrained(output_dir)


class NanoVLMGRPODataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for nanoVLM GRPO training.
    
    Each sample should have:
    - prompt: The text prompt
    - image or images: Optional image(s) (PIL Image, path, array, or list of images)
    - Any additional columns (e.g., 'answer') will be passed to reward functions
    
    Supports both single and multiple images per sample.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        prompt_column: str = "prompt",
        image_column: str = "image",
        extra_columns: Optional[list[str]] = None,
    ):
        self.dataset = dataset
        self.prompt_column = prompt_column
        self.image_column = image_column
        self.extra_columns = extra_columns or []
        
        # Auto-detect extra columns if not specified
        if not self.extra_columns:
            all_columns = dataset.column_names if hasattr(dataset, 'column_names') else list(dataset[0].keys())
            reserved = {prompt_column, image_column, "images"}  # Also reserve "images" column
            self.extra_columns = [col for col in all_columns if col not in reserved]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataset[idx]
        
        output = {
            "prompt": item[self.prompt_column],
        }
        
        # Handle both single image and multiple images
        # Check for "images" column first (multiple images), then "image" column (single image)
        images = item.get("images") or item.get(self.image_column)
        if images is not None:
            output["images"] = images
        
        # Include extra columns for reward functions
        for col in self.extra_columns:
            if col in item:
                output[col] = item[col]
        
        return output


def create_nanovlm_grpo_dataset(
    dataset: Dataset,
    prompt_column: str = "prompt",
    image_column: str = "image",
    extra_columns: Optional[list[str]] = None,
) -> NanoVLMGRPODataset:
    """
    Create a PyTorch Dataset for nanoVLM GRPO training.
    
    Args:
        dataset: HuggingFace dataset with prompts and optional images
        prompt_column: Name of the prompt column
        image_column: Name of the image column
        extra_columns: Additional columns to pass to reward functions (e.g., ['answer'])
                      If None, auto-detects all non-prompt/image columns
        
    Returns:
        NanoVLMGRPODataset ready for use with NanoVLMGRPOTrainer
    """
    return NanoVLMGRPODataset(
        dataset=dataset,
        prompt_column=prompt_column,
        image_column=image_column,
        extra_columns=extra_columns,
    )


# =============================================================================
# Example Usage
# =============================================================================
"""
Example usage of NanoVLMGRPOTrainer:

```python
import torch
from PIL import Image
from datasets import Dataset

from nanovlm.vision_language_model import VisionLanguageModel
from nanovlm.processors import get_tokenizer, get_image_processor
from nanovlm_grpo_trainer import (
    NanoVLMGRPOTrainer,
    NanoVLMGRPOConfig,
    create_nanovlm_grpo_dataset,
)


# Define a reward function
def accuracy_reward(prompts, completions, completion_ids, **kwargs):
    \"\"\"Example reward function that checks if completion contains expected answer.\"\"\"
    rewards = []
    for prompt, completion in zip(prompts, completions):
        # Your reward logic here
        # For example, check if the answer is correct
        if "correct" in completion.lower():
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


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

# 3. Prepare your dataset
# Your dataset should have columns: prompt, image (optional)
raw_data = [
    {
        "prompt": "What is 2 + 2?",
        "image": Image.open("path/to/image1.jpg").convert("RGB"),
    },
    {
        "prompt": "Describe this image.",
        "image": Image.open("path/to/image2.jpg").convert("RGB"),
    },
    # ... more samples
]
raw_dataset = Dataset.from_list(raw_data)

# 4. Create the dataset
train_dataset = create_nanovlm_grpo_dataset(
    dataset=raw_dataset,
    prompt_column="prompt",
    image_column="image",
)

# 5. Create GRPO config
grpo_config = NanoVLMGRPOConfig(
    output_dir="./nanovlm-grpo-output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    num_train_epochs=1,
    num_generations=4,  # Generate 4 completions per prompt
    max_prompt_length=512,
    max_completion_length=256,
    beta=0.1,  # KL penalty
    # Default GRPO settings (token-level importance sampling, mask truncated)
    loss_type="grpo",
    importance_sampling_level="token",
    mask_truncated_completions=True,
    temperature=1.0,
    logging_steps=10,
    save_steps=100,
)

# 6. Create the trainer
trainer = NanoVLMGRPOTrainer(
    model=model,
    reward_funcs=accuracy_reward,
    args=grpo_config,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    image_processor=image_processor,
)

# 7. Train!
trainer.train()

# 8. Save the model
model.save_pretrained("./nanovlm-grpo-finetuned")
```

For multiple reward functions:

```python
def format_reward(prompts, completions, **kwargs):
    \"\"\"Reward for well-formatted responses.\"\"\"
    rewards = []
    for completion in completions:
        # Check formatting
        if completion.strip() and not completion.startswith(" "):
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards

def length_reward(prompts, completions, **kwargs):
    \"\"\"Reward for appropriate length.\"\"\"
    rewards = []
    for completion in completions:
        length = len(completion.split())
        if 10 <= length <= 100:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards

# Use multiple reward functions
trainer = NanoVLMGRPOTrainer(
    model=model,
    reward_funcs=[accuracy_reward, format_reward, length_reward],
    args=grpo_config,
    train_dataset=train_dataset,
)
```

For GSPO (Group Sequence Policy Optimization):

```python
# GSPO config - key differences from GRPO:
# 1. importance_sampling_level="sequence" - compute importance weights at sequence level
# 2. mask_truncated_completions=False - don't mask truncated completions
# 3. loss_type="dr_grpo" - use DR-GRPO loss normalization
gspo_config = NanoVLMGRPOConfig(
    output_dir="./nanovlm-gspo-output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    num_train_epochs=1,
    num_generations=4,
    max_prompt_length=512,
    max_completion_length=256,
    beta=0.1,
    # GSPO-specific settings
    loss_type="dr_grpo",
    importance_sampling_level="sequence",
    mask_truncated_completions=False,
)

trainer = NanoVLMGRPOTrainer(
    model=model,
    reward_funcs=accuracy_reward,
    args=gspo_config,
    train_dataset=train_dataset,
)
```

For SAPO (Sigmoid-based Advantage Policy Optimization):

```python
# SAPO config - uses sigmoid smoothing for advantage weighting
sapo_config = NanoVLMGRPOConfig(
    output_dir="./nanovlm-sapo-output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    num_train_epochs=1,
    num_generations=4,
    max_prompt_length=512,
    max_completion_length=256,
    beta=0.1,
    # SAPO-specific settings
    loss_type="sapo",
    sapo_temperature_pos=1.0,  # Temperature for positive advantages
    sapo_temperature_neg=1.0,  # Temperature for negative advantages
)

trainer = NanoVLMGRPOTrainer(
    model=model,
    reward_funcs=accuracy_reward,
    args=sapo_config,
    train_dataset=train_dataset,
)
```

For DAPO/CISPO with two-sided clipping:

```python
# DAPO config with delta parameter for two-sided clipping
dapo_config = NanoVLMGRPOConfig(
    output_dir="./nanovlm-dapo-output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    num_train_epochs=1,
    num_generations=4,
    max_prompt_length=512,
    max_completion_length=256,
    beta=0.1,
    # DAPO-specific settings
    loss_type="dapo",
    epsilon=0.2,  # Lower clipping bound
    epsilon_high=0.28,  # Upper clipping bound (can be different from epsilon)
    delta=10.0,  # Optional upper bound for importance ratio (two-sided clipping)
)

trainer = NanoVLMGRPOTrainer(
    model=model,
    reward_funcs=accuracy_reward,
    args=dapo_config,
    train_dataset=train_dataset,
)
```
"""
