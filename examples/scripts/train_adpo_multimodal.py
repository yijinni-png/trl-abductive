#!/usr/bin/env python3

"""
Multimodal ADPO Training Script

This script demonstrates how to train an ADPO (Abductive Direct Preference Optimization) model
with multimodal support, where chosen and rejected prompts can contain different images.

Example usage:
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

Supported Vision-Language Models:
    - HuggingFaceTB/SmolVLM-Instruct
    - llava-hf/llava-v1.6-mistral-7b-hf  
    - meta-llama/Llama-3.2-11B-Vision-Instruct
    - HuggingFaceM4/idefics3-8b-llama

Dataset Format:
    The dataset should contain the following fields:
    {
        "response": "The expected text response",
        "chosen": "Chosen prompt text",
        "chosen_images": [PIL.Image],  # Images for chosen prompt
        "rejected": "Rejected prompt text", 
        "rejected_images": [PIL.Image],  # Different images for rejected prompt
    }
"""

import torch
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    HfArgumentParser,
    set_seed,
)
from PIL import Image
import numpy as np

from trl import ADPOConfig, ADPOTrainer, ModelConfig, get_peft_config


@dataclass
class ScriptArguments:
    """
    Arguments for multimodal ADPO training script.
    """
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    dataset_split: str = field(
        default="train",
        metadata={"help": "The split of the dataset to use."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of training examples."}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of evaluation examples."}
    )


def create_multimodal_dataset_example():
    """
    Create a small example multimodal dataset for demonstration.
    
    In practice, you would load your own dataset using load_dataset().
    """
    # Create dummy images (replace with your actual images)
    def create_dummy_image(color, size=(224, 224)):
        """Create a dummy colored image."""
        img_array = np.full((*size, 3), color, dtype=np.uint8)
        return Image.fromarray(img_array)
    
    # Example data
    data = [
        {
            "response": "This is a red object in the image.",
            "chosen": "What color is this object?",
            "chosen_images": [create_dummy_image([255, 0, 0])],  # Red image
            "rejected": "What color is this object?", 
            "rejected_images": [create_dummy_image([0, 255, 0])],  # Green image
        },
        {
            "response": "The image shows a geometric shape.",
            "chosen": "Describe what you see in the image.",
            "chosen_images": [create_dummy_image([0, 0, 255])],  # Blue image
            "rejected": "Describe what you see in the image.",
            "rejected_images": [create_dummy_image([255, 255, 0])],  # Yellow image
        },
        # Add more examples as needed
    ]
    
    return Dataset.from_list(data)


def main():
    # Parse arguments
    parser = HfArgumentParser((ScriptArguments, ADPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    
    # Set seed for reproducibility
    set_seed(training_args.seed if training_args.seed is not None else 42)
    
    # Enable multimodal mode
    training_args.multimodal_mode = True
    training_args.separate_images = True
    
    # Load dataset
    if script_args.dataset_name:
        raw_datasets = load_dataset(
            script_args.dataset_name,
            script_args.dataset_config,
            split=script_args.dataset_split
        )
    else:
        print("No dataset specified, creating example multimodal dataset...")
        raw_datasets = create_multimodal_dataset_example()
    
    # Limit training samples if specified
    if script_args.max_train_samples is not None:
        raw_datasets = raw_datasets.select(range(script_args.max_train_samples))
    
    # Print dataset info
    print(f"Dataset size: {len(raw_datasets)}")
    print("Dataset features:", raw_datasets.features)
    print("First example:", raw_datasets[0])
    
    # Load model and processor
    print(f"Loading model: {model_args.model_name_or_path}")
    
    # Model configuration
    torch_dtype = getattr(torch, model_args.torch_dtype) if model_args.torch_dtype else torch.bfloat16
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "attn_implementation": model_args.attn_implementation,
        "trust_remote_code": model_args.trust_remote_code,
    }
    
    # Load processor (handles both text and images)
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code
    )
    
    # Load model
    model = AutoModelForVision2Seq.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs
    )
    
    # Verify the model is a vision model
    print(f"Model type: {type(model)}")
    print(f"Model config: {model.config}")
    
    # Configure PEFT if needed
    peft_config = get_peft_config(model_args)
    
    # Initialize ADPO trainer
    print("Initializing multimodal ADPO trainer...")
    trainer = ADPOTrainer(
        model=model,
        args=training_args,
        train_dataset=raw_datasets,
        processing_class=processor,  # Use processor instead of tokenizer for vision models
        peft_config=peft_config,
    )
    
    # Verify trainer settings
    print(f"Multimodal mode: {training_args.multimodal_mode}")
    print(f"Separate images: {training_args.separate_images}")
    print(f"Is vision model: {trainer.is_vision_model}")
    
    # Start training
    print("Starting multimodal ADPO training...")
    trainer.train()
    
    # Save the model
    trainer.save_model(training_args.output_dir)
    if hasattr(processor, 'save_pretrained'):
        processor.save_pretrained(training_args.output_dir)
    
    print(f"Training completed! Model saved to {training_args.output_dir}")
    
    # Push to hub if specified
    if training_args.push_to_hub:
        print("Pushing model to Hugging Face Hub...")
        trainer.push_to_hub()
        if hasattr(processor, 'push_to_hub'):
            processor.push_to_hub(training_args.hub_model_id)


if __name__ == "__main__":
    main()