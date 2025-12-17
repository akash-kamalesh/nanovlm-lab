# Training Approaches

This guide provides detailed information about the three training approaches supported by NanoVLM-Lab.

---

## 🎓 Supervised Fine-Tuning (SFT)

Standard supervised learning on image-text pairs. Best for:
- Domain-specific task adaptation
- Improving base model performance
- Quick baseline establishment

### Key Features

- Uses HuggingFace Trainer framework
- Multi-LR parameter groups (MP, vision, language, LoRA)
- Mixed precision training (bf16/fp16)
- Gradient accumulation and clipping

### When to Use SFT

- You have labeled image-text pairs
- You want to adapt a pre-trained model to your domain
- You need a quick baseline before trying more advanced methods

### Example Configuration

```yaml
training_type: "sft"
model:
  model_name_or_path: "lusxvr/nanoVLM-230M-8k"
dataset:
  train_data_path: "lmms-lab/multimodal-open-r1-8k-verified"
  train_split: "train[:5%]"
  eval_data_path: null
  dataset_format: "vqa"
training:
  output_dir: "./sft_output"
  num_train_epochs: 3
  per_device_train_batch_size: 4
  learning_rate: 5e-5
```

---

## 🎯 Direct Preference Optimization (DPO)

Preference-based alignment without reward models. Best for:
- Reducing hallucinations
- Improving response quality
- Aligning with human preferences

### Key Features

- No separate reward model needed
- Efficient preference learning
- Supports multiple loss types (sigmoid, hinge)
- Configurable beta (temperature) parameter

### When to Use DPO

- You have preference pairs (chosen/rejected responses)
- You want to align the model with human preferences
- You want faster training than RLHF without a separate reward model

### Example Configuration

```yaml
training_type: "dpo"
model:
  model_name_or_path: "lusxvr/nanoVLM-230M-8k"
dataset:
  train_data_path: "HuggingFaceH4/rlaif-v_formatted"
  train_split: "train"
  eval_data_path: null
  dataset_format: "preference"
training:
  output_dir: "./dpo_output"
  num_train_epochs: 2
  per_device_train_batch_size: 4
  learning_rate: 1e-5
dpo:
  beta: 0.1
  dpo_loss_type: "sigmoid"
  label_smoothing: 0.0
```

---

## 🚀 Group Relative Policy Optimization (GRPO)

Advanced policy optimization with multiple variants. Best for:
- Complex preference structures
- Group-based reward optimization
- Research on policy gradient methods

### Key Features

- **GSPO** (Group Supervised Policy Optimization) variant
- **DAPO** (Divergence-Aware Policy Optimization) variant
- Custom reward functions
- Group-based relative scoring

### GRPO Variants

#### GRPO (Standard)
- **Use when**: You want standard group-based policy optimization
- **Configuration**:
  ```yaml
  grpo:
    loss_type: "grpo"
    importance_sampling_level: "token"
    mask_truncated_completions: true
  ```

#### GSPO (Group Supervised Policy Optimization)
- **Use when**: You want sequence-level importance sampling
- **Configuration**:
  ```yaml
  grpo:
    loss_type: "dr_grpo"
    importance_sampling_level: "sequence"
    mask_truncated_completions: false
  ```

#### DAPO (Divergence-Aware Policy Optimization)
- **Use when**: You want two-sided clipping with divergence awareness
- **Configuration**:
  ```yaml
  grpo:
    loss_type: "dapo"
    epsilon: 0.2
    epsilon_high: 0.28
    delta: 10.0
  ```

#### SAPO (Sigmoid-based Advantage Policy Optimization)
- **Use when**: You want sigmoid-based advantage weighting
- **Configuration**:
  ```yaml
  grpo:
    loss_type: "sapo"
    sapo_temperature_pos: 1.0
    sapo_temperature_neg: 1.0
  ```

### When to Use GRPO

- You have multiple completions per prompt with reward scores
- You want to optimize based on custom reward functions
- You want to experiment with different policy optimization variants

### Example Configuration

```yaml
training_type: "grpo"
model:
  model_name_or_path: "lusxvr/nanoVLM-230M-8k"
  use_lora: true
  lora_config:
    r: 16
    lora_alpha: 16
    lora_dropout: 0.1
dataset:
  train_data_path: "lmms-lab/multimodal-open-r1-8k-verified"
  train_split: "train[:5%]"
  eval_data_path: null
  dataset_format: "grpo"
  reward_functions_fn: "reward_functions.py"
training:
  output_dir: "./grpo_output"
  num_train_epochs: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  learning_rate: 5e-6
grpo:
  num_generations: 4
  max_prompt_length: 512
  max_completion_length: 256
  temperature: 1.0
  top_p: 1.0
  loss_type: "grpo"
  importance_sampling_level: "token"
  mask_truncated_completions: true
  beta: 0.1
  scale_rewards: "group"
  disable_dropout: true
  log_completions: false
```

---

## Comparison Table

| Aspect | SFT | DPO | GRPO |
|--------|-----|-----|------|
| **Input Format** | Image-text pairs | Preference pairs | Completions + rewards |
| **Reward Model** | Not needed | Not needed | Custom functions |
| **Training Speed** | Fast | Medium | Slower |
| **Memory Usage** | Low | Medium | High |
| **Complexity** | Low | Medium | High |
| **Best For** | Baseline | Alignment | Advanced optimization |

---

## Choosing Your Approach

### Start with SFT if:
- You're new to NanoVLM-Lab
- You have basic image-text pairs
- You want a quick baseline

### Move to DPO if:
- You have preference data
- You want to improve alignment
- You want faster training than RLHF

### Use GRPO if:
- You have custom reward functions
- You want fine-grained control over training
- You're doing research on policy optimization

---

## Example Notebooks

Interactive Jupyter notebooks are available for each training approach:
- `nanovlm_sft.ipynb` — Step-by-step SFT training
- `nanovlm_dpo.ipynb` — DPO preference optimization
- `nanovlm_grpo.ipynb` — GRPO with custom rewards

These notebooks provide complete examples with data loading, training, and evaluation. To begin with a notebook, just move it out of the examples folder and place it in the project's root directory.
---

For detailed configuration options, see [`configs/README.md`](../configs/README.md).
