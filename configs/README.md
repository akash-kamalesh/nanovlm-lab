# Configuration Guide

This guide explains how to configure NanoVLM-Lab for training. Start with the quick start section, then refer to the detailed sections as needed.

---

## ⚡ Quick Start (5 minutes)

### 1. Choose Your Training Type

```yaml
training_type: "sft"  # Options: "sft", "dpo", "grpo"
```

### 2. Set Your Model

```yaml
model:
  model_name_or_path: "lusxvr/nanoVLM-230M-8k"
```

### 3. Set Your Dataset

```yaml
dataset:
  train_data_path: "lmms-lab/multimodal-open-r1-8k-verified"
  train_split: "train[:5%]"
  eval_data_path: null  # Auto-split 80/20 if null
```

### 4. Set Training Parameters

```yaml
training:
  output_dir: "./output"
  num_train_epochs: 1
  per_device_train_batch_size: 4
  learning_rate: 5e-5
  bf16: true
```

### 5. Run Training

```bash
python rlvlm/main.py --config configs/your_config.yaml
```

That's it! For most use cases, these settings are all you need.

---

## 📋 Dataset Format Requirements

Before configuring your training, your dataset must be in the correct format for each training type. This section details the exact format expected by each trainer.

### SFT (Supervised Fine-Tuning) Dataset Format

**Required columns:**
- `images` — List of PIL Images (or single PIL Image)
- `texts` — List of conversation dictionaries with `user` and `assistant` keys

**Example:**
```python
{
    "images": [<PIL.Image>, <PIL.Image>],  # Can be 1-4 images
    "texts": [
        {
            "user": "What is in this image?",
            "assistant": "A cat sitting on a couch."
        },
        {
            "user": "What color is the cat?",
            "assistant": "The cat is orange."
        }
    ]
}
```

**Alternative format (ShareGPT-style conversations):**
If your dataset has a `conversations` column with role-based turns, the trainer will auto-convert it:
```python
{
    "image": <PIL.Image>,
    "conversations": [
        {"from": "human", "value": "What is this?"},
        {"from": "gpt", "value": "This is a cat."},
        {"from": "human", "value": "What color?"},
        {"from": "gpt", "value": "Orange."}
    ]
}
```

**Notes:**
- Images are automatically converted to RGB
- Maximum 4 images per example
- Each conversation turn is a user-assistant pair

---

### DPO (Direct Preference Optimization) Dataset Format

**Required columns:**
- `image` or `images` — Single PIL Image (or list of PIL Images)
- `prompt` — Text prompt/question
- `chosen` — Preferred completion/answer
- `rejected` — Non-preferred completion/answer

**Example (single image):**
```python
{
    "image": <PIL.Image>,
    "prompt": "Why are cakes usually eaten at parties?",
    "chosen": "Cakes are eaten at parties to celebrate special occasions like birthdays and anniversaries.",
    "rejected": "I don't know."
}
```

**Example (multiple images):**
```python
{
    "images": [<PIL.Image>, <PIL.Image>],  # Can be 1-4 images
    "prompt": "Compare these two images.",
    "chosen": "The first image shows a cat, the second shows a dog.",
    "rejected": "I can't tell what's in these images."
}
```

**Alternative column names:**
The trainer automatically handles these alternative names:
- `chosen_answer`, `winner_answer` → `chosen`
- `rejected_answer`, `loser_answer` → `rejected`

**Notes:**
- Supports 1-4 images per sample
- Images are automatically converted to RGB
- Prompt + chosen/rejected completions are tokenized separately
- Image tokens are preserved during truncation
- Multiple images are automatically concatenated

---

### GRPO (Group Relative Policy Optimization) Dataset Format

**Required columns:**
- `prompt` — Text prompt/question
- `image` or `images` — Optional PIL Image(s) (or path to image, or list of images)
- **Additional columns** — Any extra columns (e.g., `answer`, `expected_answer`) are passed to reward functions

**Example (single image):**
```python
{
    "image": <PIL.Image>,
    "prompt": "What is 2 + 2?",
    "answer": "4",  # Passed to reward functions
    "difficulty": "easy"  # Any extra column works
}
```

**Example (multiple images):**
```python
{
    "images": [<PIL.Image>, <PIL.Image>],  # Can be 1-4 images
    "prompt": "Compare these two images and describe the differences.",
    "answer": "The first image shows a cat, the second shows a dog.",
}
```

**Notes:**
- GRPO generates multiple completions per prompt during training
- Supports 1-4 images per sample
- Extra columns are automatically detected and passed to reward functions
- Images are optional (text-only prompts work too)
- Multiple images are automatically concatenated
- Reward functions receive: `prompts`, `completions`, `completion_ids`, and all extra columns

---

## 📋 Configuration Sections Explained

### training_type
Determines which trainer to use:
- `"sft"` — Supervised Fine-Tuning (standard training)
- `"dpo"` — Direct Preference Optimization (preference-based)
- `"grpo"` — Group Relative Policy Optimization (advanced)

### model
```yaml
model:
  model_name_or_path: "lusxvr/nanoVLM-230M-8k"  # Model to train
  use_lora: false  # Set to true for parameter-efficient training
```

### dataset
```yaml
dataset:
  train_data_path: "dataset_name"  # HuggingFace dataset ID or local path
  train_split: "train"  # Which split to use
  eval_data_path: null  # Leave null to auto-split 80/20
  dataset_format: "vqa"  # Format depends on training type
```

**Dataset formats by training type:**
- **SFT**: `"vqa"` — Image-text pairs
- **DPO**: `"preference"` — Preference pairs (chosen/rejected)
- **GRPO**: `"grpo"` — Multiple completions with rewards

### training
Core training settings:
```yaml
training:
  output_dir: "./output"  # Where to save model
  num_train_epochs: 1  # Number of training epochs
  per_device_train_batch_size: 4  # Batch size per GPU
  learning_rate: 5e-5  # Learning rate
  warmup_ratio: 0.1  # Warmup as fraction of total steps
  weight_decay: 0.01  # L2 regularization
  bf16: true  # Use bfloat16 (faster, uses less memory)
```

**If you get "out of memory" error:**
- Reduce `per_device_train_batch_size` (e.g., 4 → 2)
- Increase `gradient_accumulation_steps` (e.g., 1 → 4)

### save
```yaml
save:
  save_model_path: "./output/final_model"
  save_tokenizer: true
  save_image_processor: true
```

### logging
```yaml
logging:
  use_wandb: false  # Set to true for experiment tracking
  report_to: ["tensorboard"]
```

---

## 🎯 Training Type Specifics

### SFT (Supervised Fine-Tuning)

Use this for basic training on image-text pairs.

**Minimal config:**
```yaml
training_type: "sft"
model:
  model_name_or_path: "lusxvr/nanoVLM-230M-8k"
dataset:
  train_data_path: "your_dataset"
  dataset_format: "vqa"
training:
  output_dir: "./sft_output"
  num_train_epochs: 1
  per_device_train_batch_size: 4
  learning_rate: 5e-5
```

**Optional SFT parameters:**
```yaml
training:
  lr_mp: 5e-3  # Modality Projector learning rate (usually higher)
  lr_vision_backbone: 5e-5  # Vision encoder learning rate
  lr_language_backbone: 5e-5  # Language model learning rate
```

---

### DPO (Direct Preference Optimization)

Use this when you have preference pairs (chosen/rejected responses).

**Minimal config:**
```yaml
training_type: "dpo"
model:
  model_name_or_path: "lusxvr/nanoVLM-230M-8k"
dataset:
  train_data_path: "your_dataset"
  dataset_format: "preference"
training:
  output_dir: "./dpo_output"
  num_train_epochs: 1
  per_device_train_batch_size: 4
  learning_rate: 1e-5
dpo:
  beta: 0.1  # Higher = stronger preference signal (0.05-0.5)
  dpo_loss_type: "sigmoid"  # Loss type
```

---

### GRPO (Group Relative Policy Optimization)

Use this for advanced training with custom reward functions.

**Minimal config:**
```yaml
training_type: "grpo"
model:
  model_name_or_path: "lusxvr/nanoVLM-230M-8k"
dataset:
  train_data_path: "your_dataset"
  dataset_format: "grpo"
  reward_functions_fn: "reward_functions.py"  # REQUIRED
training:
  output_dir: "./grpo_output"
  num_train_epochs: 1
  per_device_train_batch_size: 1
  learning_rate: 5e-6
grpo:
  num_generations: 4  # Completions per prompt
  max_completion_length: 256
  loss_type: "grpo"  # Options: "grpo", "dr_grpo", "dapo", "sapo"
  beta: 0.1
```

**GRPO variants:**
- `loss_type: "grpo"` — Standard GRPO
- `loss_type: "dr_grpo"` — GSPO (Group Supervised)
- `loss_type: "dapo"` — DAPO (Divergence-Aware)
- `loss_type: "sapo"` — SAPO (Sigmoid-based)

---

## 📊 Dataset Preprocessing (Optional)

If you need to transform your data before training, create `rlvlm/dataset_preprocessing.py`:

```python
def convert_to_rgb(example):
    """Convert image to RGB."""
    if "image" in example:
        image = example["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        example["image"] = image
    return example
```

Then reference it in your config:
```yaml
dataset:
  preprocessing_fn: "dataset_preprocessing.py"
```

---

## 🎁 Example Configs

### Minimal SFT (Quick Test)
```yaml
training_type: "sft"
model:
  model_name_or_path: "lusxvr/nanoVLM-230M-8k"
dataset:
  train_data_path: "lmms-lab/multimodal-open-r1-8k-verified"
  train_split: "train[:1%]"
  dataset_format: "vqa"
training:
  output_dir: "./test_output"
  num_train_epochs: 1
  per_device_train_batch_size: 2
  learning_rate: 5e-5
  bf16: true
```

### Full SFT (Production)
```yaml
training_type: "sft"
model:
  model_name_or_path: "lusxvr/nanoVLM-230M-8k"
  use_lora: true
dataset:
  train_data_path: "lmms-lab/multimodal-open-r1-8k-verified"
  train_split: "train"
  eval_data_path: "lmms-lab/multimodal-open-r1-8k-verified"
  eval_split: "test"
  dataset_format: "vqa"
training:
  output_dir: "./sft_output"
  num_train_epochs: 3
  per_device_train_batch_size: 4
  learning_rate: 5e-5
  warmup_ratio: 0.1
  bf16: true
  eval_strategy: "steps"
  eval_steps: 500
logging:
  use_wandb: true
  wandb_project: "nanovlm-experiments"
```

### SFT with Fresh Model Initialization
```yaml
training_type: "sft"
model:
  model_name_or_path: "initialize"  # Create fresh model from config
  initialize_from_config: true
  load_backbone: true  # Load pretrained SigLIP + SmolLM2 weights
  use_lora: true
  lora_config:
    r: 16
    lora_alpha: 16
    lora_dropout: 0.1
    target_modules: ['down_proj', 'o_proj', 'k_proj', 'q_proj', 'gate_proj', 'up_proj', 'v_proj']
  vlm_config:  # Customize model architecture (optional)
    vit_model_type: "google/siglip2-base-patch16-512"
    lm_model_type: "HuggingFaceTB/SmolLM2-360M-Instruct"
    lm_tokenizer: "HuggingFaceTB/SmolLM2-360M-Instruct"
    vit_img_size: 512
    max_img_size: 2048
    mp_image_token_length: 64
dataset:
  train_data_path: "lmms-lab/multimodal-open-r1-8k-verified"
  train_split: "train[:5%]"
  dataset_format: "vqa"
training:
  output_dir: "./sft_output"
  num_train_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  learning_rate: 5e-5
  lr_mp: 5e-3
  lr_vision_backbone: 5e-5
  lr_language_backbone: 5e-5
  warmup_ratio: 0.1
  bf16: true
  eval_strategy: "steps"
  eval_steps: 500
```

### DPO (Preference Alignment)
```yaml
training_type: "dpo"
model:
  model_name_or_path: "lusxvr/nanoVLM-230M-8k"
dataset:
  train_data_path: "HuggingFaceH4/rlaif-v_formatted"
  train_split: "train"
  dataset_format: "preference"
training:
  output_dir: "./dpo_output"
  num_train_epochs: 2
  per_device_train_batch_size: 4
  learning_rate: 1e-5
  bf16: true
dpo:
  beta: 0.1
  dpo_loss_type: "sigmoid"
```

### GRPO (Custom Rewards)
```yaml
training_type: "grpo"
model:
  model_name_or_path: "lusxvr/nanoVLM-230M-8k"
  use_lora: true
dataset:
  train_data_path: "lmms-lab/multimodal-open-r1-8k-verified"
  train_split: "train[:5%]"
  dataset_format: "grpo"
  reward_functions_fn: "reward_functions.py"
training:
  output_dir: "./grpo_output"
  num_train_epochs: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  learning_rate: 5e-6
  bf16: true
grpo:
  num_generations: 4
  max_completion_length: 256
  loss_type: "grpo"
  beta: 0.1
```

---

## 🚀 Next Steps

1. Copy one of the example configs above
2. Update `model_name_or_path` with your model
3. Update `train_data_path` with your dataset
4. Run: `python rlvlm/main.py --config your_config.yaml`
5. Monitor training with TensorBoard or W&B
