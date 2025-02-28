"""
Run this file under manga-image-translator dir
"""
import asyncio
import pandas as pd
from PIL import Image
from manga_translator.manga_translator import MangaTranslator
from manga_translator.config import Config
import time
from datetime import datetime
import os
import uuid
from pathlib import Path
from typing import List, Union, Dict, Any
import cv2
import numpy as np
from itertools import product
import traceback

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

import json
import torch

torch.cuda.empty_cache()
cuda_memory_allocated_list = []
cuda_memory_reserved_list = []
IMAGE_DIR = '/mnt/sda2/werner/evaluation_pipeline_01_2025/exported_images/original_images'
OUTPUT_DIR = "./predictions_variations"
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')


def get_image_paths(image_dir: Union[str, Path]) -> List[Path]:
    """
    Retrieve all image file paths from the specified directory.
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
        
    return [
        f for f in image_dir.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def draw_detections(
    image_path: Union[str, Path],
    annotations: Dict[str, Any],
    output_path: Union[str, Path]
) -> np.ndarray:
    """
    Draw bounding boxes and labels on the image.
    
    Args:
        image_path: Path to the original image
        annotations: Dictionary containing detection annotations
        output_path: Path where the annotated image should be saved
        
    Returns:
        NumPy array containing the annotated image
        
    Raises:
        FileNotFoundError: If the input image doesn't exist
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    for obj in annotations["objects"]:
        x, y = obj["left"], obj["top"]
        w, h = obj["width"], obj["height"]
        
        # Draw bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add label
        label = obj["tag_name"]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        (text_width, text_height), _ = cv2.getTextSize(
            label, font, font_scale, thickness
        )
        
        # Draw label background
        cv2.rectangle(
            image,
            (x, y - text_height - 10),
            (x + text_width, y),
            (0, 255, 0),
            -1
        )
        
        # Draw label text
        cv2.putText(
            image,
            label,
            (x, y - 5),
            font,
            font_scale,
            (0, 0, 0),
            thickness
        )
    
    cv2.imwrite(str(output_path), image)
    return image


def get_config_variations():
    """Return a list of all possible configuration combinations."""
    # Define parameter variations # TODO: frequent changes
    # config_ur-None_rv-False_det-default_dr-False_dar-False_di-False_dgc-True_ur-3_ds-2048_tt-0.6_bt-0.7_is	
    param_variations = {
        "verbose": [True],
        "upscale_ratio": [None], # [None, 2, 4],
        "revert_upscaling": [False], # [True, False],
        "detectors": ["default"],
        "det_rotate": [False],
        "det_auto_rotate": [False],
        "det_invert": [False],
        "det_gamma_correct": [False],
        "unclip_ratio": [3],
        "detection_sizes": [2048],
        "text_thresholds": [0.6],
        "box_thresholds": [0.7],
        "inpainting_sizes": [512]
    }
    
    # Generate all combinations
    combinations = product(
        param_variations["upscale_ratio"],
        param_variations["revert_upscaling"],
        param_variations["detectors"],
        param_variations["det_rotate"],
        param_variations["det_auto_rotate"],
        param_variations["det_invert"],
        param_variations["det_gamma_correct"],
        param_variations["unclip_ratio"],
        param_variations["detection_sizes"],
        param_variations["text_thresholds"],
        param_variations["box_thresholds"],
        param_variations["inpainting_sizes"]
    )
    
    config_variations = []
    
    # Create configuration dictionary for each combination
    for (ratio, revert, detector, rotate, auto_rotate, invert, gamma_correct, unclip, det_size, text_thresh, box_thresh, inp_size) in combinations:
        config_name = f"config_ur-{ratio}_rv-{revert}_det-{detector}_dr-{rotate}_dar-{auto_rotate}_di-{invert}_dgc-{gamma_correct}_ur-{unclip}_ds-{det_size}_tt-{text_thresh}_bt-{box_thresh}_is-{inp_size}"
        
        config = {
            "name": config_name,
            "upscale": {
                "upscale_ratio": ratio,
                "revert_upscaling": revert
            },
            "detector": {
                "detector": detector,
                "det_rotate": rotate,
                "det_auto_rotate": auto_rotate,
                "det_invert": invert,
                "det_gamma_correct": gamma_correct,
                "unclip_ratio": unclip,
                "detection_size": det_size,
                "text_threshold": text_thresh,
                "box_threshold": box_thresh
            },
            "inpainting_size": inp_size
        }
        
        config_variations.append(config)
    
    print(f"Generated {len(config_variations)} configuration variations")
    return config_variations


async def process_images_with_config(translator: MangaTranslator, images: List[Path], config_variation: dict, output_dir: Path):
    """Process multiple images with a specific configuration."""
    config_output_dir = output_dir / config_variation['name']
    config_output_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = config_output_dir / "predictions.json"
    predictions_data = []
    
    # Create the full config object # TODO: frequent changes
    config = Config(
        upscale=config_variation["upscale"],
        detector=config_variation["detector"],
        ocr={"ocr": "48px"},
        translator={"translator_gen": "sugoi", "target_lang": "ENG"},
        colorizer={"colorizer": "none"},
        render={"renderer": "manga2eng", "font_size": 20},
        inpainter={"inpainter": "lama_large", "inpainting_size": config_variation["inpainting_size"]}
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    initial_reserved = torch.cuda.memory_reserved() / 1024 / 1024
    
    max_time = 0
    max_memory = 0
    
    # Process each image
    # TODO: testing only, hard coded
    # images = [Path("/mnt/sda2/werner/evaluation_pipeline_01_2025/exported_images/original_images/e215c5f7-ee32-48b4-966b-394e2631ac6d.jpg")]
    # images = [Path("/mnt/sda2/werner/evaluation_pipeline_01_2025/exported_images/original_images/36c7f325-f594-4f8e-aca5-79fffed8565f.jpg")]
    for image_path in images:
        image = Image.open(image_path)
        start_time = time.perf_counter()
        
        try:
            context = await translator.translate(image, config)
        except Exception as e:
            print(f"Exception !!! {e} - image: {image_path}")
            traceback.print_exc()
            continue
        
        end_time = time.perf_counter()
        translated_image_path = config_output_dir / f"translated_{image_path.stem}_{timestamp}.png"
        if context.result.mode == 'RGBA':
            context.result = context.result.convert('RGB')
        
        context.result.save(translated_image_path)
        current_reserved = torch.cuda.memory_reserved() / 1024 / 1024
        peak_cuda_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        peak_cuda_memory_mb += (current_reserved - initial_reserved)
        max_memory = max(max_memory, peak_cuda_memory_mb)
        max_time = max(max_time, (end_time - start_time))
        
        # Format detection results
        detection_result = {
            "image_id": image_path.stem,
            "file_name": image_path.name,
            "width": image.size[0],
            "height": image.size[1],
            "objects": []  # Initialize empty list of objects
        }

        # Only add text regions if they were detected
        if hasattr(context, 'text_regions') and context.text_regions is not None:
            detection_result["objects"] = [
                {
                    "tag_name": "text_box",
                    "left": int(box.xyxy[0]),
                    "top": int(box.xyxy[1]),
                    "width": int(box.xyxy[2] - box.xyxy[0]),
                    "height": int(box.xyxy[3] - box.xyxy[1])
                }
                for box in context.text_regions
            ]

        # Save detection visualization even if no boxes were detected
        output_image_path = config_output_dir / f"predicted_{image_path.stem}_{timestamp}.jpg"
        draw_detections(image_path, detection_result, output_image_path)
        predictions_data.append(detection_result)
    
    with open(predictions_file, "w") as f:
        json.dump(predictions_data, f, indent=4)
    
    return {
        "timestamp": timestamp,
        "config_name": config_variation["name"],
        "config_json": json.dumps(config_variation),
        "peak_cuda_memory_mb": max_memory,
        "max_inference_time_sec": round(max_time, 3),
        "num_images": len(images)
    }

async def main():
    """Main execution function."""
    try:
        output_dir = Path(OUTPUT_DIR) / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        config_params = {
            'verbose': True,
            'use_gpu': True,
            'pre_dict': None,
            'post_dict': None,
            'font_path': None,
            'kernel_size': 3
        }
        
        image_paths = get_image_paths(IMAGE_DIR)
        translator = MangaTranslator(params=config_params)
        config_variations = get_config_variations()
        df = pd.DataFrame(columns=[
            "timestamp", "config_name", "config_json", 
            "peak_cuda_memory_mb", "max_inference_time_sec", "num_images"
        ])
        
        for config_variation in config_variations: # TODO: testing only
            print(f"Testing configuration: {config_variation['name']}")
            
            result = await process_images_with_config(
                translator, image_paths, config_variation, output_dir
            )
            df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
            df.to_csv(os.path.join(output_dir, f'test_config_settings_{timestamp}.csv'), index=False)
        
        print(f"Results saved to: {os.path.join(output_dir, f'test_config_settings_{timestamp}.csv')}")
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())