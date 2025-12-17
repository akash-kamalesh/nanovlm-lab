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
NanoVLM SFT Trainer

This module implements a Supervised Fine-Tuning (SFT) Trainer for nanoVLM models,
built on top of the HuggingFace Trainer class. Adapted by Akash Kamalesh.

Key features:
1. Uses nanoVLM's forward signature: forward(input_ids, images, attention_mask, targets)
2. Custom data collator for nanoVLM's image format (VQACollator)
3. Uses nanoVLM's image processor and tokenizer
4. Multi-LR parameter groups (MP, vision backbone, language backbone, LoRA)
5. Mixed precision training (bf16/fp16)
6. Gradient accumulation and clipping
7. Warmup + cosine decay learning rate schedule
"""

import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from datasets import Dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import PreTrainedTokenizerBase, TrainingArguments
from transformers.data.data_collator import DataCollatorMixin
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import seed_worker

# nanoVLM imports
from data.datasets import VQADataset, BaseDataset
from data.collators import VQACollator
from data.processors import get_tokenizer, get_image_processor, get_image_string

# Try to import VisionLanguageModel - handle both possible locations
try:
    from models.vision_language_model import VisionLanguageModel
    from models.config import VLMConfig
except ImportError as e:
    raise ImportError("Could not find the nanovlm folder. please ensure it has been placed in the correct folder (project root).") from e
    VisionLanguageModel = None
    VLMConfig = None


__all__ = [
    "NanoVLMSFTTrainer",
    "NanoVLMSFTConfig",
    "NanoVLMSFTDataCollator",
    "create_sft_dataset",
]


@dataclass
class NanoVLMSFTConfig:
    """
    Configuration for NanoVLM SFT Trainer.
    
    This config handles both training hyperparameters and nanoVLM-specific settings.
    """
    
    # Output and checkpointing
    output_dir: str = "./nanovlm-sft-output"
    
    # Batch size and accumulation
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    
    # Learning rates for different components
    learning_rate: float = 5e-5  # Default LR (used if component-specific LRs are 0)
    lr_mp: float = 5e-3  # Modality Projector learning rate
    lr_vision_backbone: float = 5e-5  # Vision encoder learning rate (0 = frozen)
    lr_language_backbone: float = 5e-5  # Language model learning rate (0 = frozen)
    lr_lora: float = 5e-4  # LoRA adapter learning rate (if using PEFT)
    
    # Training duration
    num_train_epochs: int = 1
    max_steps: int = -1  # -1 means use num_train_epochs
    
    # Learning rate schedule
    warmup_ratio: float = 0.03
    warmup_steps: int = 0  # Overrides warmup_ratio if > 0
    lr_scheduler_type: str = "cosine"
    
    # Regularization
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    
    # Mixed precision
    bf16: bool = True
    fp16: bool = False
    
    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 2
    eval_steps: int = 500
    eval_strategy: str = "steps"  # "steps", "epoch", or "no"
    
    # Data processing
    max_seq_length: int = 8192
    max_images_per_example: int = 4  # Filter samples with more images
    
    # Misc
    seed: int = 42
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    remove_unused_columns: bool = False
    
    # Model compilation
    compile_model: bool = False
    
    # Resume from checkpoint
    resume_from_checkpoint: Optional[str] = None
    
    def to_training_arguments(self) -> TrainingArguments:
        """Convert to HuggingFace TrainingArguments."""
        return TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_train_epochs,
            max_steps=self.max_steps,
            warmup_ratio=self.warmup_ratio,
            warmup_steps=self.warmup_steps,
            lr_scheduler_type=self.lr_scheduler_type,
            weight_decay=self.weight_decay,
            max_grad_norm=self.max_grad_norm,
            bf16=self.bf16,
            fp16=self.fp16,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            eval_steps=self.eval_steps,
            eval_strategy=self.eval_strategy,
            seed=self.seed,
            dataloader_num_workers=self.dataloader_num_workers,
            dataloader_pin_memory=self.dataloader_pin_memory,
            remove_unused_columns=self.remove_unused_columns,
            report_to="none",  # Can be overridden
        )


class NanoVLMSFTDataCollator(DataCollatorMixin):
    """
    Data collator for nanoVLM SFT training.
    
    Wraps the existing VQACollator to work with HuggingFace Trainer.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 8192,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vqa_collator = VQACollator(tokenizer, max_length)
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate a batch of features."""
        return self.vqa_collator(features)


class NanoVLMSFTTrainer(Trainer):
    """
    Supervised Fine-Tuning Trainer for nanoVLM models.
    
    This trainer extends HuggingFace's Trainer to work with nanoVLM's
    vision-language architecture, handling:
    - Custom forward pass with images
    - Multi-LR parameter groups
    - PEFT/LoRA integration
    
    Args:
        model: The nanoVLM model to train (VisionLanguageModel or PEFT-wrapped)
        args: NanoVLMSFTConfig or TrainingArguments
        train_dataset: Training dataset (should be a VQADataset or compatible)
        eval_dataset: Evaluation dataset
        tokenizer: nanoVLM tokenizer
        data_collator: Data collator (default: NanoVLMSFTDataCollator)
        callbacks: Training callbacks
        optimizers: Optimizer and scheduler tuple (optional, for custom optimization)
    """
    
    def __init__(
        self,
        model: nn.Module,
        args: Optional[Union[NanoVLMSFTConfig, TrainingArguments]] = None,
        train_dataset: Optional[TorchDataset] = None,
        eval_dataset: Optional[TorchDataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        data_collator: Optional[DataCollatorMixin] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ):
        # Store SFT config if provided
        if isinstance(args, NanoVLMSFTConfig):
            self.sft_config = args
            training_args = args.to_training_arguments()
        else:
            self.sft_config = NanoVLMSFTConfig()
            training_args = args if args is not None else self.sft_config.to_training_arguments()
        
        # Get tokenizer from model if not provided
        if tokenizer is None:
            if hasattr(model, 'tokenizer'):
                tokenizer = model.tokenizer
            elif hasattr(model, 'model') and hasattr(model.model, 'tokenizer'):
                # PEFT wrapped model
                tokenizer = model.model.tokenizer
            else:
                raise ValueError("Tokenizer must be provided or model must have a tokenizer attribute")
        
        # Create data collator if not provided
        if data_collator is None:
            data_collator = NanoVLMSFTDataCollator(
                tokenizer=tokenizer,
                max_length=self.sft_config.max_seq_length,
            )
        
        # Initialize parent Trainer
        super().__init__(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            processing_class=tokenizer,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        
        # Store reference to base model (unwrap PEFT if needed)
        self._base_model = self._get_base_model(model)
        
        # Compile model if requested
        if self.sft_config.compile_model:
            self.model = torch.compile(self.model)
    
    def _get_base_model(self, model: nn.Module) -> nn.Module:
        """Get the base nanoVLM model, unwrapping PEFT if necessary."""
        if hasattr(model, 'model'):
            # PEFT wrapped model
            return model.model
        return model
    
    def create_optimizer(self):
        """
        Create optimizer with multi-LR parameter groups for nanoVLM components.
        
        This allows different learning rates for:
        - Modality Projector (MP)
        - Vision encoder
        - Language model backbone
        - LoRA adapters (if using PEFT)
        """
        if self.optimizer is not None:
            return self.optimizer
        
        base_model = self._base_model
        param_groups = []
        
        # Check if using PEFT
        is_peft = hasattr(self.model, 'peft_config')
        
        # Modality Projector
        if hasattr(base_model, 'MP'):
            if self.sft_config.lr_mp > 0:
                param_groups.append({
                    "params": list(base_model.MP.parameters()),
                    "lr": self.sft_config.lr_mp,
                    "name": "modality_projector",
                })
            else:
                for p in base_model.MP.parameters():
                    p.requires_grad = False
        
        # Vision encoder
        if hasattr(base_model, 'vision_encoder'):
            if self.sft_config.lr_vision_backbone > 0:
                param_groups.append({
                    "params": list(base_model.vision_encoder.parameters()),
                    "lr": self.sft_config.lr_vision_backbone,
                    "name": "vision_encoder",
                })
            else:
                for p in base_model.vision_encoder.parameters():
                    p.requires_grad = False
        
        # Language model / decoder
        if is_peft:
            # Separate LoRA params from base LM params
            lora_params = [
                p for name, p in self.model.named_parameters()
                if "lora_" in name and p.requires_grad
            ]
            if lora_params and self.sft_config.lr_lora > 0:
                param_groups.append({
                    "params": lora_params,
                    "lr": self.sft_config.lr_lora,
                    "name": "lora_adapters",
                })
            
            # Base LM params (excluding LoRA)
            if hasattr(base_model, 'decoder'):
                base_lm_params = [
                    p for name, p in base_model.decoder.named_parameters()
                    if "lora_" not in name and p.requires_grad
                ]
                if base_lm_params and self.sft_config.lr_language_backbone > 0:
                    param_groups.append({
                        "params": base_lm_params,
                        "lr": self.sft_config.lr_language_backbone,
                        "name": "language_backbone",
                    })
                elif self.sft_config.lr_language_backbone == 0:
                    for p in base_lm_params:
                        p.requires_grad = False
        else:
            # No PEFT - just language backbone
            if hasattr(base_model, 'decoder'):
                if self.sft_config.lr_language_backbone > 0:
                    param_groups.append({
                        "params": list(base_model.decoder.parameters()),
                        "lr": self.sft_config.lr_language_backbone,
                        "name": "language_backbone",
                    })
                else:
                    for p in base_model.decoder.parameters():
                        p.requires_grad = False
        
        # Fallback: if no param groups, use all trainable params with default LR
        if not param_groups:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            if trainable_params:
                param_groups.append({
                    "params": trainable_params,
                    "lr": self.args.learning_rate,
                    "name": "all_params",
                })
        
        # Log parameter groups
        for group in param_groups:
            num_params = sum(p.numel() for p in group["params"])
            print(f"  {group.get('name', 'unnamed')}: {num_params:,} params, lr={group['lr']}")
        
        # Create AdamW optimizer
        self.optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.args.weight_decay,
        )
        
        return self.optimizer
    
    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Compute the training loss using nanoVLM's forward pass.
        
        nanoVLM forward signature: forward(input_ids, images, attention_mask, targets)
        Returns: (hidden_states, loss)
        """
        # Extract inputs
        input_ids = inputs.get("input_ids")
        images = inputs.get("images")
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")
        
        # Move tensors to device
        device = self.args.device
        if input_ids is not None:
            input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)
        
        # Handle images - they may be a list of tensors
        if images is not None:
            if isinstance(images, list):
                # Keep as list, model handles it
                pass
            elif isinstance(images, torch.Tensor):
                images = images.to(device)
        
        # Forward pass through nanoVLM
        # nanoVLM returns (hidden_states, loss) when targets are provided
        outputs = model(
            input_ids=input_ids,
            images=images,
            attention_mask=attention_mask,
            targets=labels,
        )
        
        # Extract loss
        if isinstance(outputs, tuple):
            hidden_states, loss = outputs
        else:
            # Fallback if model returns differently
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs
            hidden_states = outputs
        
        if return_outputs:
            return loss, {"hidden_states": hidden_states}
        return loss
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step.
        """
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        
        return (loss, None, None)
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Save the model, handling PEFT models appropriately."""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if PEFT model
        if hasattr(self.model, 'save_pretrained'):
            # PEFT model or HF model with save_pretrained
            self.model.save_pretrained(output_dir)
        else:
            # Standard PyTorch save
            torch.save(
                self.model.state_dict(),
                os.path.join(output_dir, "pytorch_model.bin")
            )
        
        # Save tokenizer
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        
        # Save training args
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def create_sft_dataset(
    hf_dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    image_processor: Callable,
    mp_image_token_length: int = 64,
    image_column: str = "image",
    images_column: str = "images",
    text_column: str = "texts",
    max_images_per_example: int = 4,
    convert_to_rgb: bool = True,
) -> VQADataset:
    """
    Convert a HuggingFace dataset to a nanoVLM SFT dataset.
    
    This function takes a HuggingFace dataset and prepares it for use with
    the NanoVLMSFTTrainer by wrapping it in a VQADataset.
    
    Expected dataset format:
    - images: List of PIL Images or single PIL Image
    - texts: List of dicts with 'user' and 'assistant' keys for conversation turns
    
    Example:
        {
            "images": [<PIL.Image>],
            "texts": [
                {"user": "What is in this image?", "assistant": "A cat sitting on a couch."}
            ]
        }
    
    Args:
        hf_dataset: HuggingFace Dataset object
        tokenizer: nanoVLM tokenizer
        image_processor: nanoVLM image processor
        mp_image_token_length: Number of tokens per image patch
        image_column: Column name for single image
        images_column: Column name for multiple images
        text_column: Column name for conversation texts
        max_images_per_example: Maximum images allowed per example
        convert_to_rgb: Whether to convert images to RGB
    
    Returns:
        VQADataset ready for training
    """
    
    def preprocess_example(example):
        """Preprocess a single example to match VQADataset expected format."""
        # Handle images
        if images_column in example and example[images_column] is not None:
            images = example[images_column]
            if not isinstance(images, list):
                images = [images]
        elif image_column in example and example[image_column] is not None:
            images = [example[image_column]]
        else:
            images = []
        
        # Convert to RGB if needed
        if convert_to_rgb:
            processed_images = []
            for img in images:
                if isinstance(img, Image.Image):
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                processed_images.append(img)
            images = processed_images
        
        # Filter by max images
        if len(images) > max_images_per_example:
            return None  # Will be filtered out
        
        # Handle texts - ensure proper format
        texts = example.get(text_column, [])
        if not isinstance(texts, list):
            texts = [texts]
        
        return {
            "images": images,
            "texts": texts,
        }
    
    # Apply preprocessing
    processed_dataset = hf_dataset.map(
        preprocess_example,
        remove_columns=hf_dataset.column_names,
        num_proc=4,
    )
    
    # Filter out None values (examples that exceeded max_images)
    processed_dataset = processed_dataset.filter(
        lambda x: x is not None and x.get("texts") is not None and len(x.get("texts", [])) > 0
    )
    
    # Wrap in VQADataset
    return VQADataset(
        dataset=processed_dataset,
        tokenizer=tokenizer,
        image_processor=image_processor,
        mp_image_token_length=mp_image_token_length,
    )


def create_sft_dataset_from_conversations(
    hf_dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    image_processor: Callable,
    mp_image_token_length: int = 64,
    image_column: str = "image",
    conversation_column: str = "conversations",
    user_role: str = "human",
    assistant_role: str = "gpt",
    max_images_per_example: int = 4,
) -> VQADataset:
    """
    Convert a HuggingFace dataset with conversation format to nanoVLM SFT dataset.
    
    This handles datasets with a "conversations" column containing role-based turns,
    like ShareGPT format:
    [
        {"from": "human", "value": "What is this?"},
        {"from": "gpt", "value": "This is a cat."}
    ]
    
    Args:
        hf_dataset: HuggingFace Dataset object
        tokenizer: nanoVLM tokenizer
        image_processor: nanoVLM image processor
        mp_image_token_length: Number of tokens per image patch
        image_column: Column name for image(s)
        conversation_column: Column name for conversations
        user_role: Role identifier for user messages
        assistant_role: Role identifier for assistant messages
        max_images_per_example: Maximum images allowed per example
    
    Returns:
        VQADataset ready for training
    """
    
    def convert_conversation(example):
        """Convert conversation format to texts format."""
        conversations = example.get(conversation_column, [])
        
        # Handle images
        images = example.get(image_column)
        if images is None:
            images = []
        elif not isinstance(images, list):
            images = [images]
        
        # Convert images to RGB
        processed_images = []
        for img in images:
            if isinstance(img, Image.Image):
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                processed_images.append(img)
        
        if len(processed_images) > max_images_per_example:
            return {"images": [], "texts": []}  # Will be filtered
        
        # Convert conversations to texts format
        texts = []
        i = 0
        while i < len(conversations) - 1:
            turn = conversations[i]
            next_turn = conversations[i + 1]
            
            # Check for user-assistant pair
            from_key = "from" if "from" in turn else "role"
            value_key = "value" if "value" in turn else "content"
            
            if turn.get(from_key) == user_role and next_turn.get(from_key) == assistant_role:
                texts.append({
                    "user": turn.get(value_key, ""),
                    "assistant": next_turn.get(value_key, ""),
                })
                i += 2
            else:
                i += 1
        
        return {
            "images": processed_images,
            "texts": texts,
        }
    
    # Apply conversion
    processed_dataset = hf_dataset.map(
        convert_conversation,
        remove_columns=hf_dataset.column_names,
        num_proc=4,
    )
    
    # Filter empty examples
    processed_dataset = processed_dataset.filter(
        lambda x: len(x.get("texts", [])) > 0
    )
    
    # Wrap in VQADataset
    return VQADataset(
        dataset=processed_dataset,
        tokenizer=tokenizer,
        image_processor=image_processor,
        mp_image_token_length=mp_image_token_length,
    )


# Utility function for getting cosine LR with warmup (for manual LR scheduling if needed)
def get_cosine_schedule_with_warmup_lr(
    current_step: int,
    max_lr: float,
    max_steps: int,
    warmup_ratio: float = 0.03,
) -> float:
    """
    Calculate learning rate using cosine schedule with linear warmup.
    
    Args:
        current_step: Current training step
        max_lr: Maximum learning rate
        max_steps: Total training steps
        warmup_ratio: Fraction of steps for warmup
    
    Returns:
        Learning rate for current step
    """
    min_lr = max_lr * 0.1
    warmup_steps = int(max_steps * warmup_ratio)
    
    if current_step < warmup_steps:
        # Linear warmup
        return max_lr * (current_step + 1) / warmup_steps
    elif current_step > max_steps:
        return min_lr
    else:
        # Cosine decay
        decay_ratio = (current_step - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)
