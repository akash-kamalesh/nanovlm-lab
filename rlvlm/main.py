"""
NanoVLM-Lab CLI Entry Point

This module provides the main entry point for running NanoVLM training experiments
via command-line interface with YAML configuration files.

Usage:
    python -m rlvlm.main --config configs/sft_config.yaml
    python rlvlm/main.py --config configs/sft_config.yaml
"""

import argparse
import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import yaml
from datasets import load_dataset, Dataset

# Add project root and nanovlm to path
# This allows:
# 1. Importing nanovlm package from project root
# 2. NanoVLM's internal imports (e.g., from models.utils) to work
project_root = Path(__file__).parent.parent
nanovlm_root = project_root / "nanovlm"

if str(nanovlm_root) not in sys.path:
    sys.path.insert(0, str(nanovlm_root))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Change working directory to nanovlm so relative imports work
os.chdir(nanovlm_root)

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dictionary containing configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.is_absolute():
        config_path = project_root / config_path
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")
    
    logger.info(f"Loaded config from: {config_path}")
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate that required fields are present in config.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If required fields are missing
    """
    required_fields = ['training_type', 'model', 'dataset', 'training']
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")
    
    training_type = config.get('training_type', '').lower()
    if training_type not in ['sft', 'dpo', 'grpo']:
        raise ValueError(f"Invalid training_type: {training_type}. Must be 'sft', 'dpo', or 'grpo'")
    
    logger.info(f"Config validation passed for training_type: {training_type}")


def setup_logging(config: Dict[str, Any]) -> None:
    """
    Setup logging configuration.
    
    Args:
        config: Configuration dictionary
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_info()


def load_preprocessing_functions(preprocessing_fn_path: Optional[str]) -> List[Callable]:
    """
    Load preprocessing functions from a Python file.
    
    Args:
        preprocessing_fn_path: Path to Python file containing preprocessing functions
                              Can be relative (looked up in rlvlm dir) or absolute
        
    Returns:
        List of callable preprocessing functions found in the file (in definition order)
    """
    if not preprocessing_fn_path:
        logger.info("No preprocessing functions specified")
        return []
    
    preprocessing_fn_path = Path(preprocessing_fn_path)
    
    if not preprocessing_fn_path.is_absolute():
        rlvlm_dir = Path(__file__).parent
        preprocessing_fn_path = rlvlm_dir / preprocessing_fn_path
    
    if not preprocessing_fn_path.exists():
        logger.warning(f"Preprocessing file not found: {preprocessing_fn_path}")
        return []
    
    logger.info(f"Loading preprocessing functions from: {preprocessing_fn_path}")
    
    spec = importlib.util.spec_from_file_location("preprocessing_module", preprocessing_fn_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Use __dict__ to preserve definition order instead of dir() which sorts alphabetically
    preprocessing_functions = []
    for name, obj in module.__dict__.items():
        if not name.startswith('_'):
            if callable(obj) and not isinstance(obj, type):
                preprocessing_functions.append((name, obj))
                logger.info(f"  Found preprocessing function: {name}")
    
    return preprocessing_functions


def load_hf_dataset(
    dataset_id: str,
    split: str = "train",
    max_samples: Optional[int] = None
) -> Dataset:
    """
    Load a HuggingFace dataset.
    
    Args:
        dataset_id: HuggingFace dataset repo ID (e.g., "username/dataset-name")
        split: Dataset split to load (default: "train")
        max_samples: Maximum number of samples to load (optional)
        
    Returns:
        Loaded HuggingFace dataset
    """
    logger.info(f"Loading dataset: {dataset_id} (split: {split})")
    
    dataset = load_dataset(dataset_id, split=split)
    
    if max_samples is not None and max_samples > 0:
        logger.info(f"Limiting dataset to {max_samples} samples")
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    logger.info(f"Loaded {len(dataset)} samples")
    return dataset


def apply_preprocessing(
    dataset: Dataset,
    preprocessing_functions: List[tuple]
) -> Dataset:
    """
    Apply preprocessing functions to a dataset sequentially.
    
    Args:
        dataset: HuggingFace dataset to preprocess
        preprocessing_functions: List of (name, function) tuples
        
    Returns:
        Preprocessed dataset
    """
    if not preprocessing_functions:
        logger.info("No preprocessing functions to apply")
        return dataset
    
    for func_name, func in preprocessing_functions:
        logger.info(f"Applying preprocessing: {func_name}")
        
        # Check if function is designed for batched processing
        if func_name.startswith("batch_"):
            dataset = dataset.map(func, batched=True, num_proc=4)
        else:
            dataset = dataset.map(func)
    
    return dataset


def load_and_preprocess_datasets(
    config: Dict[str, Any]
) -> tuple[Dataset, Dataset]:
    """
    Load and preprocess train and eval datasets.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    dataset_config = config.get('dataset', {})
    
    # Get dataset paths and splits
    train_data_path = dataset_config.get('train_data_path')
    train_split = dataset_config.get('train_split', 'train')
    eval_data_path = dataset_config.get('eval_data_path')
    eval_split = dataset_config.get('eval_split', 'test')
    max_samples = dataset_config.get('max_samples')
    preprocessing_fn = dataset_config.get('preprocessing_fn')
    rename_columns = dataset_config.get('rename_columns', {})
    
    if not train_data_path:
        raise ValueError("dataset.train_data_path is required")
    
    # Load preprocessing functions
    preprocessing_functions = load_preprocessing_functions(preprocessing_fn)
    
    logger.info("=" * 80)
    logger.info("Loading Training Dataset")
    logger.info("=" * 80)
    train_dataset = load_hf_dataset(train_data_path, train_split, max_samples)
    
    # Rename columns if specified
    if rename_columns:
        logger.info(f"Renaming columns: {rename_columns}")
        for old_name, new_name in rename_columns.items():
            if old_name in train_dataset.column_names:
                train_dataset = train_dataset.rename_column(old_name, new_name)
                logger.info(f"  Renamed '{old_name}' -> '{new_name}'")
    
    # Apply preprocessing to training dataset
    train_dataset = apply_preprocessing(train_dataset, preprocessing_functions)
    
    if eval_data_path:
        logger.info("=" * 80)
        logger.info("Loading Evaluation Dataset")
        logger.info("=" * 80)
        eval_dataset = load_hf_dataset(eval_data_path, eval_split, max_samples)
        
        if rename_columns:
            logger.info(f"Renaming columns: {rename_columns}")
            for old_name, new_name in rename_columns.items():
                if old_name in eval_dataset.column_names:
                    eval_dataset = eval_dataset.rename_column(old_name, new_name)
                    logger.info(f"  Renamed '{old_name}' -> '{new_name}'")
        
        eval_dataset = apply_preprocessing(eval_dataset, preprocessing_functions)
    else:
        logger.info("=" * 80)
        logger.info("Creating Evaluation Dataset from Training Data")
        logger.info("=" * 80)
        logger.info("Splitting training data: 80% train, 20% eval")
        split_dataset = train_dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']
        logger.info(f"Train dataset: {len(train_dataset)} samples")
        logger.info(f"Eval dataset: {len(eval_dataset)} samples")
    
    return train_dataset, eval_dataset


def run_sft_training(config: Dict[str, Any]) -> None:
    """
    Run SFT (Supervised Fine-Tuning) training.
    
    Args:
        config: Configuration dictionary
    """
    logger.info("=" * 80)
    logger.info("Starting SFT (Supervised Fine-Tuning) Training")
    logger.info("=" * 80)
    
    try:
        from nanovlm.models.vision_language_model import VisionLanguageModel
        from nanovlm.data.processors import get_tokenizer, get_image_processor
        from nanovlm_sft_trainer import NanoVLMSFTTrainer, NanoVLMSFTConfig, create_sft_dataset
    except ImportError as e:
        logger.error(f"Failed to import SFT trainer components: {e}")
        raise
    
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    save_config = config.get('save', {})
    logging_config = config.get('logging', {})
    
    model_path = model_config.get('model_name_or_path')
    initialize_from_config = model_config.get('initialize_from_config', False)
    
    if not model_path:
        raise ValueError("model.model_name_or_path is required")
    
    logger.info("=" * 80)
    logger.info("Loading Model")
    logger.info("=" * 80)
    
    if model_path.lower() == "initialize" or initialize_from_config:
        logger.info("Creating fresh nanoVLM model from config")
        # TODO: Implement model initialization from config
        raise NotImplementedError("Model initialization from config not yet implemented")
    else:
        logger.info(f"Loading pre-trained model from: {model_path}")
        model = VisionLanguageModel.from_pretrained(model_path)
    
    model = model.to(training_config.get('device', 'cuda'))
    
    logger.info("=" * 80)
    logger.info("Model Configuration")
    logger.info("=" * 80)
    logger.info(f"  lm_tokenizer: {model.cfg.lm_tokenizer}")
    logger.info(f"  vlm_extra_tokens: {model.cfg.vlm_extra_tokens}")
    logger.info(f"  lm_chat_template: {model.cfg.lm_chat_template}")
    logger.info(f"  max_img_size: {model.cfg.max_img_size}")
    logger.info(f"  vit_img_size: {model.cfg.vit_img_size}")
    logger.info(f"  resize_to_max_side_len: {getattr(model.cfg, 'resize_to_max_side_len', 'NOT SET')}")
    
    # Apply LoRA if enabled
    use_lora = model_config.get('use_lora', False)
    if use_lora:
        logger.info("=" * 80)
        logger.info("Applying LoRA Configuration")
        logger.info("=" * 80)
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            logger.error("PEFT library not installed. Install with: pip install peft")
            raise
        
        lora_config_dict = model_config.get('lora_config', {})
        peft_config = LoraConfig(
            r=int(lora_config_dict.get('r', 16)),
            lora_alpha=int(lora_config_dict.get('lora_alpha', 16)),
            lora_dropout=float(lora_config_dict.get('lora_dropout', 0.1)),
            target_modules=lora_config_dict.get('target_modules', ['down_proj', 'o_proj', 'k_proj', 'q_proj', 'gate_proj', 'up_proj', 'v_proj']),
            use_dora=lora_config_dict.get('use_dora', False),
            init_lora_weights=lora_config_dict.get('init_lora_weights', 'gaussian'),
        )
        logger.info(f"LoRA config: r={peft_config.r}, lora_alpha={peft_config.lora_alpha}, lora_dropout={peft_config.lora_dropout}")
        logger.info(f"Target modules: {peft_config.target_modules}")
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    logger.info("Loading tokenizer and image processor")
    tokenizer = get_tokenizer(
        model.cfg.lm_tokenizer,
        model.cfg.vlm_extra_tokens,
        model.cfg.lm_chat_template
    )
    image_processor = get_image_processor(
        model.cfg.max_img_size,
        model.cfg.vit_img_size,
        getattr(model.cfg, 'resize_to_max_side_len', False)
    )
    
    logger.info("=" * 80)
    logger.info("Loading and Preprocessing Datasets")
    logger.info("=" * 80)
    train_dataset, eval_dataset = load_and_preprocess_datasets(config)
    
    logger.info("=" * 80)
    logger.info("Converting Datasets to SFT Format")
    logger.info("=" * 80)
    logger.info("Converting training dataset to SFT format")
    train_dataset = create_sft_dataset(train_dataset, tokenizer, image_processor)
    
    logger.info("Converting evaluation dataset to SFT format")
    eval_dataset = create_sft_dataset(eval_dataset, tokenizer, image_processor)
    
    logger.info("=" * 80)
    logger.info("Creating SFT Configuration")
    logger.info("=" * 80)
    sft_config = NanoVLMSFTConfig(
        output_dir=training_config.get('output_dir', './sft_output'),
        per_device_train_batch_size=int(training_config.get('per_device_train_batch_size', 4)),
        per_device_eval_batch_size=int(training_config.get('per_device_eval_batch_size', 4)),
        gradient_accumulation_steps=int(training_config.get('gradient_accumulation_steps', 1)),
        learning_rate=float(training_config.get('learning_rate', 5e-5)),
        lr_mp=float(training_config.get('lr_mp', 5e-3)),
        lr_vision_backbone=float(training_config.get('lr_vision_backbone', 5e-5)),
        lr_language_backbone=float(training_config.get('lr_language_backbone', 5e-5)),
        lr_lora=float(training_config.get('lr_lora', 5e-4)),
        num_train_epochs=int(training_config.get('num_train_epochs', 3)),
        max_steps=int(training_config.get('max_steps', -1)),
        warmup_ratio=float(training_config.get('warmup_ratio', 0.1)),
        weight_decay=float(training_config.get('weight_decay', 0.01)),
        max_grad_norm=float(training_config.get('max_grad_norm', 1.0)),
        lr_scheduler_type=training_config.get('lr_scheduler_type', 'cosine'),
        bf16=training_config.get('bf16', True),
        fp16=training_config.get('fp16', False),
        save_steps=int(training_config.get('save_steps', 500)),
        eval_strategy=training_config.get('eval_strategy', 'steps'),
        eval_steps=int(training_config.get('eval_steps', 500)),
        logging_steps=int(training_config.get('logging_steps', 100)),
        save_total_limit=int(training_config.get('save_total_limit', 3)),
        seed=int(training_config.get('seed', 42)),
    )
    
    logger.info("=" * 80)
    logger.info("Creating SFT Trainer")
    logger.info("=" * 80)
    trainer = NanoVLMSFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=None,
    )
    
    logger.info("=" * 80)
    logger.info("Starting Training")
    logger.info("=" * 80)
    trainer.train()
    
    logger.info("=" * 80)
    logger.info("Saving Model")
    logger.info("=" * 80)
    save_path = save_config.get('save_model_path', './sft_output/final_model')
    logger.info(f"Saving final model to: {save_path}")
    trainer.save_model(save_path)
    
    if save_config.get('save_tokenizer', True):
        logger.info("Saving tokenizer")
        tokenizer.save_pretrained(save_path)
    
    if save_config.get('save_image_processor', True):
        logger.info("Saving image processor")
        image_processor.save_pretrained(save_path)
    
    logger.info("=" * 80)
    logger.info("SFT Training Completed Successfully")
    logger.info("=" * 80)


def run_dpo_training(config: Dict[str, Any]) -> None:
    """
    Run DPO (Direct Preference Optimization) training.
    
    Args:
        config: Configuration dictionary
    """
    logger.info("=" * 80)
    logger.info("Starting DPO (Direct Preference Optimization) Training")
    logger.info("=" * 80)
    
    try:
        from nanovlm.models.vision_language_model import VisionLanguageModel
        from nanovlm.data.processors import get_tokenizer, get_image_processor
        from nanovlm_dpo_trainer import NanoVLMDPOTrainer, create_nanovlm_dpo_dataset
    except ImportError as e:
        logger.error(f"Failed to import DPO trainer components: {e}")
        raise
    
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    dpo_config = config.get('dpo', {})
    save_config = config.get('save', {})
    logging_config = config.get('logging', {})
    
    model_path = model_config.get('model_name_or_path')
    if not model_path:
        raise ValueError("model.model_name_or_path is required")
    
    logger.info("=" * 80)
    logger.info("Loading Model")
    logger.info("=" * 80)
    logger.info(f"Loading pre-trained model from: {model_path}")
    model = VisionLanguageModel.from_pretrained(model_path)
    model = model.to(training_config.get('device', 'cuda'))
    
    logger.info("=" * 80)
    logger.info("Model Configuration")
    logger.info("=" * 80)
    logger.info(f"  lm_tokenizer: {model.cfg.lm_tokenizer}")
    logger.info(f"  vlm_extra_tokens: {model.cfg.vlm_extra_tokens}")
    logger.info(f"  lm_chat_template: {model.cfg.lm_chat_template}")
    logger.info(f"  max_img_size: {model.cfg.max_img_size}")
    logger.info(f"  vit_img_size: {model.cfg.vit_img_size}")
    logger.info(f"  resize_to_max_side_len: {getattr(model.cfg, 'resize_to_max_side_len', 'NOT SET')}")
    
    use_lora = model_config.get('use_lora', False)
    if use_lora:
        logger.info("=" * 80)
        logger.info("Applying LoRA Configuration")
        logger.info("=" * 80)
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            logger.error("PEFT library not installed. Install with: pip install peft")
            raise
        
        lora_config_dict = model_config.get('lora_config', {})
        peft_config = LoraConfig(
            r=int(lora_config_dict.get('r', 16)),
            lora_alpha=int(lora_config_dict.get('lora_alpha', 16)),
            lora_dropout=float(lora_config_dict.get('lora_dropout', 0.1)),
            target_modules=lora_config_dict.get('target_modules', ['down_proj', 'o_proj', 'k_proj', 'q_proj', 'gate_proj', 'up_proj', 'v_proj']),
            use_dora=lora_config_dict.get('use_dora', False),
            init_lora_weights=lora_config_dict.get('init_lora_weights', 'gaussian'),
        )
        logger.info(f"LoRA config: r={peft_config.r}, lora_alpha={peft_config.lora_alpha}, lora_dropout={peft_config.lora_dropout}")
        logger.info(f"Target modules: {peft_config.target_modules}")
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    logger.info("Loading tokenizer and image processor")
    tokenizer = get_tokenizer(
        model.cfg.lm_tokenizer,
        model.cfg.vlm_extra_tokens,
        model.cfg.lm_chat_template
    )
    image_processor = get_image_processor(
        model.cfg.max_img_size,
        model.cfg.vit_img_size,
        getattr(model.cfg, 'resize_to_max_side_len', False)
    )
    
    logger.info("=" * 80)
    logger.info("Loading and Preprocessing Datasets")
    logger.info("=" * 80)
    train_dataset, eval_dataset = load_and_preprocess_datasets(config)
    
    logger.info("=" * 80)
    logger.info("Converting Datasets to DPO Format")
    logger.info("=" * 80)
    logger.info("Converting training dataset to DPO format")
    train_dataset = create_nanovlm_dpo_dataset(
        dataset=train_dataset,
        image_processor=image_processor,
        tokenizer=tokenizer,
        model_cfg=model.cfg,
        max_prompt_length=None,
        max_completion_length=1024,
    )
    
    logger.info("Converting evaluation dataset to DPO format")
    eval_dataset = create_nanovlm_dpo_dataset(
        dataset=eval_dataset,
        image_processor=image_processor,
        tokenizer=tokenizer,
        model_cfg=model.cfg,
        max_prompt_length=None,
        max_completion_length=1024,
    )
    
    logger.info("=" * 80)
    logger.info("Creating DPO Configuration")
    logger.info("=" * 80)
    from transformers import TrainingArguments
    
    dpo_training_args = TrainingArguments(
        output_dir=training_config.get('output_dir', './dpo_output'),
        per_device_train_batch_size=int(training_config.get('per_device_train_batch_size', 4)),
        per_device_eval_batch_size=int(training_config.get('per_device_eval_batch_size', 4)),
        gradient_accumulation_steps=int(training_config.get('gradient_accumulation_steps', 1)),
        learning_rate=float(training_config.get('learning_rate', 5e-5)),
        num_train_epochs=int(training_config.get('num_train_epochs', 3)),
        max_steps=int(training_config.get('max_steps', -1)),
        warmup_ratio=float(training_config.get('warmup_ratio', 0.1)),
        weight_decay=float(training_config.get('weight_decay', 0.01)),
        max_grad_norm=float(training_config.get('max_grad_norm', 1.0)),
        lr_scheduler_type=training_config.get('lr_scheduler_type', 'cosine'),
        bf16=training_config.get('bf16', True),
        fp16=training_config.get('fp16', False),
        save_strategy=training_config.get('save_strategy', 'steps'),
        save_steps=int(training_config.get('save_steps', 500)),
        eval_strategy=training_config.get('eval_strategy', 'steps'),
        eval_steps=int(training_config.get('eval_steps', 500)),
        logging_steps=int(training_config.get('logging_steps', 100)),
        save_total_limit=int(training_config.get('save_total_limit', 3)),
        seed=int(training_config.get('seed', 42)),
        report_to=logging_config.get('report_to', ['tensorboard']),
    )
    
    logger.info("=" * 80)
    logger.info("Creating DPO Trainer")
    logger.info("=" * 80)
    trainer = NanoVLMDPOTrainer(
        model=model,
        args=dpo_training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        image_processor=image_processor,
        beta=dpo_config.get('beta', 0.1),
        loss_type=dpo_config.get('dpo_loss_type', 'sigmoid'),
        label_smoothing=dpo_config.get('label_smoothing', 0.0),
    )
    
    logger.info("=" * 80)
    logger.info("Starting Training")
    logger.info("=" * 80)
    trainer.train()
    
    logger.info("=" * 80)
    logger.info("Saving Model")
    logger.info("=" * 80)
    save_path = save_config.get('save_model_path', './dpo_output/final_model')
    logger.info(f"Saving final model to: {save_path}")
    trainer.save_model(save_path)
    
    if save_config.get('save_tokenizer', True):
        logger.info("Saving tokenizer")
        tokenizer.save_pretrained(save_path)
    
    if save_config.get('save_image_processor', True):
        logger.info("Saving image processor")
        image_processor.save_pretrained(save_path)
    
    logger.info("=" * 80)
    logger.info("DPO Training Completed Successfully")
    logger.info("=" * 80)


def load_reward_functions(reward_functions_fn_path: Optional[str]) -> List[Callable]:
    """
    Load reward functions from a Python file.
    
    Args:
        reward_functions_fn_path: Path to Python file containing reward functions
                                 Can be relative (looked up in rlvlm dir) or absolute
        
    Returns:
        List of callable reward functions found in the file
    """
    if not reward_functions_fn_path:
        logger.warning("No reward functions specified")
        return []
    
    reward_functions_fn_path = Path(reward_functions_fn_path)
    
    if not reward_functions_fn_path.is_absolute():
        rlvlm_dir = Path(__file__).parent
        reward_functions_fn_path = rlvlm_dir / reward_functions_fn_path
    
    if not reward_functions_fn_path.exists():
        logger.warning(f"Reward functions file not found: {reward_functions_fn_path}")
        return []
    
    logger.info(f"Loading reward functions from: {reward_functions_fn_path}")
    
    spec = importlib.util.spec_from_file_location("reward_module", reward_functions_fn_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Extract all callable functions (excluding private/magic methods and type hints)
    reward_functions = []
    for name, obj in module.__dict__.items():
        if not name.startswith('_'):
            if callable(obj) and not isinstance(obj, type):
                if hasattr(obj, '__module__') and obj.__module__ == module.__name__:
                    reward_functions.append(obj)
                    logger.info(f"  Found reward function: {name}")
    
    return reward_functions


def run_grpo_training(config: Dict[str, Any]) -> None:
    """
    Run GRPO (Group Relative Policy Optimization) training.
    
    Args:
        config: Configuration dictionary
    """
    logger.info("=" * 80)
    logger.info("Starting GRPO (Group Relative Policy Optimization) Training")
    logger.info("=" * 80)
    
    try:
        from nanovlm.models.vision_language_model import VisionLanguageModel
        from nanovlm.data.processors import get_tokenizer, get_image_processor
        from nanovlm_grpo_trainer import NanoVLMGRPOTrainer, create_nanovlm_grpo_dataset
    except ImportError as e:
        logger.error(f"Failed to import GRPO trainer components: {e}")
        raise
    
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    grpo_config = config.get('grpo', {})
    save_config = config.get('save', {})
    logging_config = config.get('logging', {})
    
    model_path = model_config.get('model_name_or_path')
    if not model_path:
        raise ValueError("model.model_name_or_path is required")
    
    logger.info("=" * 80)
    logger.info("Loading Model")
    logger.info("=" * 80)
    logger.info(f"Loading pre-trained model from: {model_path}")
    model = VisionLanguageModel.from_pretrained(model_path)
    model = model.to(training_config.get('device', 'cuda'))
    
    logger.info("=" * 80)
    logger.info("Model Configuration")
    logger.info("=" * 80)
    logger.info(f"  lm_tokenizer: {model.cfg.lm_tokenizer}")
    logger.info(f"  vlm_extra_tokens: {model.cfg.vlm_extra_tokens}")
    logger.info(f"  lm_chat_template: {model.cfg.lm_chat_template}")
    logger.info(f"  max_img_size: {model.cfg.max_img_size}")
    logger.info(f"  vit_img_size: {model.cfg.vit_img_size}")
    logger.info(f"  resize_to_max_side_len: {getattr(model.cfg, 'resize_to_max_side_len', 'NOT SET')}")
    
    use_lora = model_config.get('use_lora', False)
    if use_lora:
        logger.info("=" * 80)
        logger.info("Applying LoRA Configuration")
        logger.info("=" * 80)
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            logger.error("PEFT library not installed. Install with: pip install peft")
            raise
        
        lora_config_dict = model_config.get('lora_config', {})
        peft_config = LoraConfig(
            r=int(lora_config_dict.get('r', 16)),
            lora_alpha=int(lora_config_dict.get('lora_alpha', 16)),
            lora_dropout=float(lora_config_dict.get('lora_dropout', 0.1)),
            target_modules=lora_config_dict.get('target_modules', ['down_proj', 'o_proj', 'k_proj', 'q_proj', 'gate_proj', 'up_proj', 'v_proj']),
            use_dora=lora_config_dict.get('use_dora', False),
            init_lora_weights=lora_config_dict.get('init_lora_weights', 'gaussian'),
        )
        logger.info(f"LoRA config: r={peft_config.r}, lora_alpha={peft_config.lora_alpha}, lora_dropout={peft_config.lora_dropout}")
        logger.info(f"Target modules: {peft_config.target_modules}")
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    logger.info("Loading tokenizer and image processor")
    tokenizer = get_tokenizer(
        model.cfg.lm_tokenizer,
        model.cfg.vlm_extra_tokens,
        model.cfg.lm_chat_template
    )
    image_processor = get_image_processor(
        model.cfg.max_img_size,
        model.cfg.vit_img_size,
        getattr(model.cfg, 'resize_to_max_side_len', False)
    )
    
    logger.info("=" * 80)
    logger.info("Loading and Preprocessing Datasets")
    logger.info("=" * 80)
    train_dataset, eval_dataset = load_and_preprocess_datasets(config)
    
    logger.info("=" * 80)
    logger.info("Loading Reward Functions")
    logger.info("=" * 80)
    dataset_config = config.get('dataset', {})
    reward_functions_fn = dataset_config.get('reward_functions_fn')
    reward_functions = load_reward_functions(reward_functions_fn)
    
    if not reward_functions:
        logger.warning("No reward functions loaded. GRPO training requires reward functions.")
        raise ValueError("reward_functions_fn must be specified in dataset config for GRPO training")
    
    logger.info("=" * 80)
    logger.info("Converting Datasets to GRPO Format")
    logger.info("=" * 80)
    logger.info("Converting training dataset to GRPO format")
    train_dataset = create_nanovlm_grpo_dataset(
        dataset=train_dataset,
        prompt_column="prompt",
        image_column="image",
    )
    
    logger.info("Converting evaluation dataset to GRPO format")
    eval_dataset = create_nanovlm_grpo_dataset(
        dataset=eval_dataset,
        prompt_column="prompt",
        image_column="image",
    )
    
    logger.info("=" * 80)
    logger.info("Creating GRPO Configuration")
    logger.info("=" * 80)
    from nanovlm_grpo_trainer import NanoVLMGRPOConfig
    
    grpo_training_config = NanoVLMGRPOConfig(
        output_dir=training_config.get('output_dir', './grpo_output'),
        per_device_train_batch_size=int(training_config.get('per_device_train_batch_size', 1)),
        per_device_eval_batch_size=int(training_config.get('per_device_eval_batch_size', 1)),
        gradient_accumulation_steps=int(training_config.get('gradient_accumulation_steps', 4)),
        learning_rate=float(training_config.get('learning_rate', 5e-6)),
        num_train_epochs=int(training_config.get('num_train_epochs', 1)),
        max_steps=int(training_config.get('max_steps', -1)),
        warmup_ratio=float(training_config.get('warmup_ratio', 0.1)),
        weight_decay=float(training_config.get('weight_decay', 0.01)),
        logging_steps=int(training_config.get('logging_steps', 10)),
        save_steps=int(training_config.get('save_steps', 100)),
        eval_steps=int(training_config.get('eval_steps', 50)),
        save_strategy=training_config.get('save_strategy', 'steps'),
        eval_strategy=training_config.get('eval_strategy', 'steps'),
        seed=int(training_config.get('seed', 42)),
        bf16=training_config.get('bf16', True),
        fp16=training_config.get('fp16', False),
        # GRPO-specific parameters
        beta=float(grpo_config.get('beta', 0.1)),
        num_generations=int(grpo_config.get('num_generations', 4)),
        max_prompt_length=int(grpo_config.get('max_prompt_length', 512)),
        max_completion_length=int(grpo_config.get('max_completion_length', 256)),
        temperature=float(grpo_config.get('temperature', 1.0)),
        top_p=float(grpo_config.get('top_p', 1.0)),
        top_k=int(grpo_config.get('top_k', 0)),
        loss_type=grpo_config.get('loss_type', 'grpo'),
        epsilon=float(grpo_config.get('epsilon', 0.2)),
        epsilon_high=float(grpo_config.get('epsilon_high', 0.2)) if grpo_config.get('epsilon_high') is not None else None,
        delta=float(grpo_config.get('delta', 10.0)) if grpo_config.get('delta') is not None else None,
        scale_rewards=grpo_config.get('scale_rewards', 'group'),
        sapo_temperature_pos=float(grpo_config.get('sapo_temperature_pos', 1.0)) if grpo_config.get('sapo_temperature_pos') is not None else None,
        sapo_temperature_neg=float(grpo_config.get('sapo_temperature_neg', 1.0)) if grpo_config.get('sapo_temperature_neg') is not None else None,
        importance_sampling_level=grpo_config.get('importance_sampling_level', 'token'),
        mask_truncated_completions=grpo_config.get('mask_truncated_completions', True),
        disable_dropout=grpo_config.get('disable_dropout', True),
        log_completions=grpo_config.get('log_completions', False),
        num_completions_to_print=int(grpo_config.get('num_completions_to_print', 5)),
    )
    
    logger.info("=" * 80)
    logger.info("Creating GRPO Trainer")
    logger.info("=" * 80)
    trainer = NanoVLMGRPOTrainer(
        model=model,
        reward_funcs=reward_functions,
        args=grpo_training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        image_processor=image_processor,
    )
    
    logger.info("=" * 80)
    logger.info("Starting Training")
    logger.info("=" * 80)
    trainer.train()
    
    # Save final model
    logger.info("=" * 80)
    logger.info("Saving Model")
    logger.info("=" * 80)
    save_path = save_config.get('save_model_path', './grpo_output/final_model')
    logger.info(f"Saving final model to: {save_path}")
    trainer.save_model(save_path)
    
    if save_config.get('save_tokenizer', True):
        logger.info("Saving tokenizer")
        tokenizer.save_pretrained(save_path)
    
    if save_config.get('save_image_processor', True):
        logger.info("Saving image processor")
        image_processor.save_pretrained(save_path)
    
    logger.info("=" * 80)
    logger.info("GRPO Training Completed Successfully")
    logger.info("=" * 80)


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='NanoVLM-Lab: Training Framework for Vision-Language Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rlvlm/main.py --config configs/sft_config.yaml
  python rlvlm/main.py --config configs/dpo_config.yaml
  python rlvlm/main.py --config configs/grpo_config.yaml
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging({})
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load and validate config
        config = load_config(args.config)
        validate_config(config)
        
        # Route to appropriate trainer
        training_type = config.get('training_type', '').lower()
        
        if training_type == 'sft':
            run_sft_training(config)
        elif training_type == 'dpo':
            run_dpo_training(config)
        elif training_type == 'grpo':
            run_grpo_training(config)
        else:
            raise ValueError(f"Unknown training_type: {training_type}")
        
        logger.info("All training completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
