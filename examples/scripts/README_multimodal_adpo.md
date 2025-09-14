# Multimodal ADPO Training

This directory contains examples for training ADPO (Abductive Direct Preference Optimization) models with multimodal support, where chosen and rejected prompts can contain different images.

## Overview

Multimodal ADPO extends the standard ADPO training to support vision-language models. Instead of comparing text-only prompts, it compares multimodal contexts:

- **Standard ADPO**: P(response | chosen_text) vs P(response | rejected_text)  
- **Multimodal ADPO**: P(response | chosen_text + chosen_images) vs P(response | rejected_text + rejected_images)

This enables the model to learn more robust reasoning patterns by understanding how different visual contexts affect the same textual response.

## Supported Models

The following vision-language models are supported:

- **SmolVLM**: `HuggingFaceTB/SmolVLM-Instruct`
- **LLaVA**: `llava-hf/llava-v1.6-mistral-7b-hf`
- **Llama Vision**: `meta-llama/Llama-3.2-11B-Vision-Instruct`
- **Idefics3**: `HuggingFaceM4/idefics3-8b-llama`

## Dataset Format

Your multimodal ADPO dataset should have the following structure:

```python
{
    "response": "The expected text response",
    "chosen": "Chosen prompt text",
    "chosen_images": [PIL.Image],  # Images for chosen prompt
    "rejected": "Rejected prompt text", 
    "rejected_images": [PIL.Image],  # Different images for rejected prompt
}
```

### Example Dataset Entry

```python
{
    "response": "This is a red apple.",
    "chosen": "What fruit is shown in this image?",
    "chosen_images": [<PIL.Image of red apple>],
    "rejected": "What fruit is shown in this image?",
    "rejected_images": [<PIL.Image of green apple>],
}
```

## Usage

### Basic Training Command

```bash
python train_adpo_multimodal.py \
    --model_name_or_path HuggingFaceTB/SmolVLM-Instruct \
    --dataset_name your_multimodal_dataset \
    --output_dir ./adpo-multimodal-model \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-7 \
    --num_train_epochs 3 \
    --multimodal_mode \
    --separate_images \
    --bf16 \
    --gradient_checkpointing
```

### Configuration Parameters

The following parameters control multimodal ADPO behavior:

- `--multimodal_mode`: Enable multimodal ADPO training
- `--separate_images`: Use separate images for chosen and rejected prompts (default: True)
- `--max_image_size`: Maximum image resolution (optional)
- `--image_processor_kwargs`: Additional processor arguments (optional)

### Memory Optimization

For large vision models, consider these optimizations:

```bash
# Use gradient checkpointing to save memory
--gradient_checkpointing

# Use mixed precision training
--bf16  # or --fp16

# Reduce batch size and increase gradient accumulation
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16

# Use DeepSpeed Zero for multi-GPU training
--deepspeed configs/deepspeed_zero3.yaml
```

## Creating Your Dataset

### Option 1: From Hugging Face Datasets

```python
from datasets import load_dataset

# Load your dataset
dataset = load_dataset("your_username/your_multimodal_adpo_dataset")

# Ensure it has the required fields
required_fields = ["response", "chosen", "chosen_images", "rejected", "rejected_images"]
assert all(field in dataset.column_names for field in required_fields)
```

### Option 2: Create Custom Dataset

```python
from datasets import Dataset
from PIL import Image

def create_multimodal_dataset(data_path):
    examples = []
    
    # Load your data
    for item in your_data:
        example = {
            "response": item["response"],
            "chosen": item["chosen_text"],
            "chosen_images": [Image.open(img_path) for img_path in item["chosen_image_paths"]],
            "rejected": item["rejected_text"],
            "rejected_images": [Image.open(img_path) for img_path in item["rejected_image_paths"]],
        }
        examples.append(example)
    
    return Dataset.from_list(examples)
```

## Training Tips

### 1. Data Quality
- Ensure chosen and rejected prompts have meaningful differences
- Use high-quality, relevant images
- Verify image-text alignment

### 2. Model Selection
- SmolVLM is recommended for quick experimentation
- LLaVA for established performance
- Llama Vision for latest capabilities

### 3. Hyperparameters
- Start with learning rate `5e-7` for vision models
- Use smaller batch sizes due to memory requirements
- Monitor gradient norms and adjust accordingly

### 4. Evaluation
- Implement custom evaluation metrics for multimodal reasoning
- Test on held-out multimodal examples
- Compare against text-only baseline

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
--per_device_train_batch_size 1
--gradient_accumulation_steps 16

# Enable gradient checkpointing
--gradient_checkpointing

# Use DeepSpeed
--deepspeed configs/deepspeed_zero3.yaml
```

**2. Image Loading Errors**
- Ensure images are in RGB format
- Check image file paths are correct
- Verify PIL.Image compatibility

**3. Model Loading Issues**
- Update transformers to latest version
- Check model compatibility with your CUDA version
- Verify trust_remote_code settings

### Debug Mode

To debug your dataset and model setup:

```bash
python train_adpo_multimodal.py \
    --max_train_samples 10 \
    --max_steps 5 \
    --logging_steps 1 \
    --save_steps 1000000  # Disable saving during debug
```

## Advanced Usage

### Custom Image Processing

```python
# In your config
--image_processor_kwargs '{"size": {"height": 336, "width": 336}, "do_rescale": true}'
```

### PEFT Integration

```python
# Use LoRA for efficient training
--use_peft \
--lora_r 16 \
--lora_alpha 32 \
--lora_target_modules vision_tower,mm_projector
```

### Multi-GPU Training

```bash
# Using accelerate
accelerate launch --config_file multi_gpu_config.yaml train_adpo_multimodal.py [args]

# Using DeepSpeed
python train_adpo_multimodal.py \
    --deepspeed configs/deepspeed_zero2.yaml \
    [other args]
```

## Performance Benchmarks

Expected training speeds on different hardware:

| Model | GPU | Batch Size | Speed (samples/sec) |
|-------|-----|------------|-------------------|
| SmolVLM | RTX 4090 | 1 | 0.8 |
| LLaVA-7B | RTX 4090 | 1 | 0.6 |
| SmolVLM | H100 | 4 | 3.2 |
| LLaVA-7B | H100 | 4 | 2.4 |

## Citation

If you use multimodal ADPO in your research, please cite:

```bibtex
@article{multimodal_adpo_2024,
    title={Multimodal Abductive Direct Preference Optimization},
    author={Your Name},
    journal={arXiv preprint},
    year={2024}
}
```