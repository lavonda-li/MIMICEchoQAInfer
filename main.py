"""
MIMICEchoQA Inference Script
Run visual question answering on echocardiogram images using MedGemma and LLaVA-Med models.
"""

import json
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText


# Paths
DATA_DIR = Path("/home/lavondali/MIMICEchoQAInfer/Data/mimic-iv-ext-echoqa/1.0.0/MIMICEchoQA")
DICOM_DIR = Path("/home/lavondali/mount-folder/MIMIC-Echo-IV")
QA_JSON = DATA_DIR / "MIMICEchoQA.json"
OUTPUT_DIR = Path("/home/lavondali/MIMICEchoQAInfer/outputs")


def load_qa_data(json_path: Path) -> list:
    """Load the QA dataset."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def video_path_to_dicom_path(video_path: str) -> Path:
    """
    Convert video path from JSON to DICOM path.
    Example: mimic-iv-echo/0.1/files/p13/p13265185/s97751020/97751020_0026.mp4
           -> /home/lavondali/mount-folder/MIMIC-Echo-IV/p13/p13265185/s97751020/97751020_0026.dcm
    """
    parts = video_path.split("/")
    # parts: ['mimic-iv-echo', '0.1', 'files', 'p13', 'p13265185', 's97751020', '97751020_0026.mp4']
    patient_prefix = parts[3]  # p13
    patient_id = parts[4]      # p13265185
    study_id = parts[5]        # s97751020
    filename = parts[6].replace(".mp4", ".dcm")  # 97751020_0026.dcm

    return DICOM_DIR / patient_prefix / patient_id / study_id / filename


def extract_frame_from_dicom(dicom_path: Path, frame_idx: int = 0) -> Image.Image:
    """
    Extract a single frame from a DICOM file.
    For multi-frame DICOM (videos), extracts the middle frame by default.
    """
    dcm = pydicom.dcmread(dicom_path)
    pixel_array = dcm.pixel_array

    # Handle multi-frame DICOM
    if len(pixel_array.shape) == 3:
        # Multi-frame: shape is (frames, height, width)
        num_frames = pixel_array.shape[0]
        if frame_idx == -1:
            # Use middle frame
            frame_idx = num_frames // 2
        frame = pixel_array[frame_idx]
    elif len(pixel_array.shape) == 4:
        # Multi-frame RGB: shape is (frames, height, width, channels)
        num_frames = pixel_array.shape[0]
        if frame_idx == -1:
            frame_idx = num_frames // 2
        frame = pixel_array[frame_idx]
    else:
        # Single frame
        frame = pixel_array

    # Normalize to 0-255 if needed
    if frame.max() > 255:
        frame = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8)
    else:
        frame = frame.astype(np.uint8)

    # Convert to RGB if grayscale
    if len(frame.shape) == 2:
        frame = np.stack([frame] * 3, axis=-1)

    return Image.fromarray(frame)


def extract_multiple_frames(dicom_path: Path, num_frames: int = 5) -> list:
    """Extract multiple frames evenly spaced throughout the video."""
    dcm = pydicom.dcmread(dicom_path)
    pixel_array = dcm.pixel_array

    # Get total frames
    if len(pixel_array.shape) >= 3:
        total_frames = pixel_array.shape[0]
    else:
        return [extract_frame_from_dicom(dicom_path, 0)]

    # Get evenly spaced frame indices
    if total_frames <= num_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

    frames = []
    for idx in indices:
        frames.append(extract_frame_from_dicom(dicom_path, idx))

    return frames


def format_question_with_options(sample: dict) -> str:
    """Format the question with multiple choice options."""
    question = sample["question"]
    options = []
    for opt in ["A", "B", "C", "D"]:
        opt_text = sample.get(f"option_{opt}", "")
        if opt_text:
            options.append(f"{opt}. {opt_text}")

    if options:
        return f"{question}\n\nOptions:\n" + "\n".join(options)
    return question


class MedGemmaInference:
    """MedGemma model inference class."""

    def __init__(self, model_id: str = "google/medgemma-4b-it", device: str = "cuda"):
        self.device = device
        self.model_id = model_id
        print(f"Loading MedGemma model: {model_id}")

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        print("MedGemma model loaded successfully")

    def predict(self, image: Image.Image, question: str) -> str:
        """Run inference on a single image with a question."""
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an expert cardiologist analyzing echocardiogram images. Answer the question based on the echocardiogram image provided. Give only the answer option letter and the answer text."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
            generation = generation[0][input_len:]

        return self.processor.decode(generation, skip_special_tokens=True)


class LLavaMedInference:
    """LLaVA-Med model inference class using the llava library."""

    def __init__(self, model_id: str = "microsoft/llava-med-v1.5-mistral-7b", device: str = "cuda"):
        self.device = device
        self.model_id = model_id
        print(f"Loading LLaVA-Med model: {model_id}")

        # Import llava components
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates

        model_name = get_model_name_from_path(model_id)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_id,
            model_base=None,
            model_name=model_name,
            device_map="auto"
        )
        # Convert model to float16 for memory efficiency
        self.model = self.model.half()

        self.conv_mode = "mistral_instruct"
        self.process_images = process_images
        self.tokenizer_image_token = tokenizer_image_token
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.conv_templates = conv_templates

        print("LLaVA-Med model loaded successfully")

    def predict(self, image: Image.Image, question: str) -> str:
        """Run inference on a single image with a question."""
        # Process image
        image_tensor = self.process_images([image], self.image_processor, self.model.config)
        image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        # Create conversation
        conv = self.conv_templates[self.conv_mode].copy()
        prompt = f"You are an expert cardiologist analyzing echocardiogram images. {question}\n\nAnswer with only the option letter and answer text."
        conv.append_message(conv.roles[0], self.DEFAULT_IMAGE_TOKEN + "\n" + prompt)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        # Tokenize
        input_ids = self.tokenizer_image_token(
            prompt_text, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.model.device)

        # Generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                max_new_tokens=100,
                use_cache=True
            )

        output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Extract only the assistant's response
        if conv.roles[1] in output:
            output = output.split(conv.roles[1])[-1].strip()
        return output


def run_inference(model_name: str, num_samples: int = None, use_middle_frame: bool = True):
    """
    Run inference on the MIMICEchoQA dataset.

    Args:
        model_name: "medgemma" or "llava-med"
        num_samples: Number of samples to process (None for all)
        use_middle_frame: If True, use middle frame; if False, use multiple frames
    """
    # Load QA data
    print(f"Loading QA data from {QA_JSON}")
    qa_data = load_qa_data(QA_JSON)
    print(f"Loaded {len(qa_data)} QA samples")

    if num_samples:
        qa_data = qa_data[:num_samples]
        print(f"Processing {num_samples} samples")

    # Initialize model
    if model_name == "medgemma":
        model = MedGemmaInference()
    elif model_name == "llava-med":
        model = LLavaMedInference()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run inference
    results = []
    for sample in tqdm(qa_data, desc=f"Running {model_name} inference"):
        video_path = sample["videos"][0]
        dicom_path = video_path_to_dicom_path(video_path)

        if not dicom_path.exists():
            print(f"DICOM file not found: {dicom_path}")
            results.append({
                "messages_id": sample["messages_id"],
                "prediction": "ERROR: DICOM not found",
                "ground_truth": sample["answer"],
                "correct_option": sample["correct_option"],
                "error": True
            })
            continue

        try:
            # Extract frame(s)
            if use_middle_frame:
                image = extract_frame_from_dicom(dicom_path, frame_idx=-1)
            else:
                # For multiple frames, use the middle one for now
                # Future: could aggregate predictions across frames
                image = extract_frame_from_dicom(dicom_path, frame_idx=-1)

            # Format question
            question = format_question_with_options(sample)

            # Run prediction
            prediction = model.predict(image, question)

            results.append({
                "messages_id": sample["messages_id"],
                "question": sample["question"],
                "prediction": prediction,
                "ground_truth": sample["answer"],
                "correct_option": sample["correct_option"],
                "structure": sample["structure"],
                "view": sample["view"],
                "error": False
            })

        except Exception as e:
            print(f"Error processing {sample['messages_id']}: {e}")
            results.append({
                "messages_id": sample["messages_id"],
                "prediction": f"ERROR: {str(e)}",
                "ground_truth": sample["answer"],
                "correct_option": sample["correct_option"],
                "error": True
            })

    # Save results
    output_file = OUTPUT_DIR / f"{model_name}_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")

    # Calculate accuracy
    correct = sum(1 for r in results if not r["error"] and r["correct_option"] in r["prediction"])
    total = sum(1 for r in results if not r["error"])
    if total > 0:
        print(f"Accuracy: {correct}/{total} = {correct/total:.2%}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run MIMICEchoQA inference")
    parser.add_argument("--model", type=str, choices=["medgemma", "llava-med", "both"],
                        default="medgemma", help="Model to use for inference")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples to process (default: all)")
    parser.add_argument("--use-middle-frame", action="store_true", default=True,
                        help="Use middle frame from video (default: True)")

    args = parser.parse_args()

    if args.model == "both":
        print("Running inference with MedGemma...")
        run_inference("medgemma", args.num_samples, args.use_middle_frame)
        print("\nRunning inference with LLaVA-Med...")
        run_inference("llava-med", args.num_samples, args.use_middle_frame)
    else:
        run_inference(args.model, args.num_samples, args.use_middle_frame)


if __name__ == "__main__":
    main()
