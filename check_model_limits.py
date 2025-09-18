#!/usr/bin/env python3
"""
Check the actual maximum grid dimensions supported by Qwen2.5-VL model.
This will help us understand what limits to apply to image_grid_thw values.
"""

from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

def main():
    print("=== CHECKING QWEN2.5-VL MODEL/PROCESSOR LIMITS ===")

    # Load the model and processor
    processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct', trust_remote_code=True, use_fast=False)

    # Don't load the full model on CPU to save memory, just get config
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct', trust_remote_code=True)
        print("✅ Loaded model config successfully")
    except Exception as e:
        print(f"❌ Could not load model config: {e}")
        config = None

    print("\n=== PROCESSOR CONFIGURATION ===")
    # Check processor configuration
    if hasattr(processor, 'image_processor'):
        img_proc = processor.image_processor
        print("Image processor attributes:")
        for attr in sorted(dir(img_proc)):
            if any(keyword in attr.lower() for keyword in ['max', 'size', 'grid', 'patch', 'height', 'width', 'resolution']):
                try:
                    value = getattr(img_proc, attr)
                    if not callable(value) and not attr.startswith('_'):
                        print(f"  {attr}: {value}")
                except:
                    pass

    print("\n=== MODEL CONFIGURATION ===")
    if config:
        print("Model config attributes:")
        for attr in sorted(dir(config)):
            if any(keyword in attr.lower() for keyword in ['max', 'grid', 'pos', 'embed', 'vision', 'image', 'patch']):
                try:
                    value = getattr(config, attr)
                    if not callable(value) and not attr.startswith('_'):
                        print(f"  {attr}: {value}")
                except:
                    pass

    print("\n=== VISION-SPECIFIC CONFIGURATION ===")
    if config and hasattr(config, 'vision_config'):
        vision_config = config.vision_config
        print("Vision config attributes:")
        for attr in sorted(dir(vision_config)):
            if any(keyword in attr.lower() for keyword in ['max', 'grid', 'pos', 'embed', 'patch', 'height', 'width']):
                try:
                    value = getattr(vision_config, attr)
                    if not callable(value) and not attr.startswith('_'):
                        print(f"  {attr}: {value}")
                except:
                    pass

    print("\n=== TESTING DIFFERENT IMAGE_GRID_THW VALUES ===")
    # Test what happens with different grid values
    test_values = [
        [1, 30, 40],   # Your current typical values
        [1, 50, 50],   # Moderate size
        [1, 60, 80],   # Large size that might cause issues
        [1, 32, 32],   # Common vision transformer size
    ]

    for t, h, w in test_values:
        grid_thw = torch.tensor([[t, h, w]], dtype=torch.long)
        print(f"Testing grid_thw {[t, h, w]}:")

        # Try to see if this would cause indexing issues
        # Note: We can't fully test without loading the model, but we can check basic constraints
        total_patches = h * w
        print(f"  Total patches: {total_patches}")

        if h > 100 or w > 100:
            print(f"  ⚠️  Very large grid dimensions - likely to cause issues")
        elif h > 50 or w > 50:
            print(f"  ⚠️  Large grid dimensions - might cause issues")
        else:
            print(f"  ✅ Reasonable grid dimensions")

if __name__ == "__main__":
    main()