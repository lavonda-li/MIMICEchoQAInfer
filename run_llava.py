"""
LLaVA-Med Inference Script
Run visual question answering on echocardiogram images using LLaVA-Med model.
Requires separate conda environment with transformers==4.36.2
"""

import json
import argparse
from pathlib import Path

import numpy as np
import pydicom
import torch
from PIL import Image
from tqdm import tqdm

# LLaVA imports
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates


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
    """Convert video path from JSON to DICOM path."""
    parts = video_path.split("/")
    patient_prefix = parts[3]
    patient_id = parts[4]
    study_id = parts[5]
    filename = parts[6].replace(".mp4", ".dcm")
    return DICOM_DIR / patient_prefix / patient_id / study_id / filename


def extract_frame_from_dicom(dicom_path: Path, frame_idx: int = -1) -> Image.Image:
    """Extract a single frame from a DICOM file."""
    dcm = pydicom.dcmread(dicom_path)
    pixel_array = dcm.pixel_array

    if len(pixel_array.shape) == 3:
        num_frames = pixel_array.shape[0]
        if frame_idx == -1:
            frame_idx = num_frames // 2
        frame = pixel_array[frame_idx]
    elif len(pixel_array.shape) == 4:
        num_frames = pixel_array.shape[0]
        if frame_idx == -1:
            frame_idx = num_frames // 2
        frame = pixel_array[frame_idx]
    else:
        frame = pixel_array

    if frame.max() > 255:
        frame = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8)
    else:
        frame = frame.astype(np.uint8)

    if len(frame.shape) == 2:
        frame = np.stack([frame] * 3, axis=-1)

    return Image.fromarray(frame)


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


class LLavaMedInference:
    """LLaVA-Med model inference class."""

    def __init__(self, model_id: str = "microsoft/llava-med-v1.5-mistral-7b"):
        print(f"Loading LLaVA-Med model: {model_id}")

        model_name = get_model_name_from_path(model_id)
        # Use 4-bit quantization to fit in T4 GPU memory
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_id,
            model_base=None,
            model_name=model_name,
            load_8bit=False,
            load_4bit=True,
            device="cuda"
        )

        self.conv_mode = "mistral_instruct"
        self.process_images = process_images
        self.tokenizer_image_token = tokenizer_image_token
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.conv_templates = conv_templates

        print("LLaVA-Med model loaded successfully")

    def predict(self, image: Image.Image, question: str) -> str:
        """Run inference on a single image with a question."""
        image_tensor = self.process_images([image], self.image_processor, self.model.config)
        image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        conv = self.conv_templates[self.conv_mode].copy()
        prompt = f"You are an expert cardiologist analyzing echocardiogram images. {question}\n\nAnswer with only the option letter and answer text."
        conv.append_message(conv.roles[0], self.DEFAULT_IMAGE_TOKEN + "\n" + prompt)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        input_ids = self.tokenizer_image_token(
            prompt_text, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                max_new_tokens=100,
                use_cache=True
            )

        output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if conv.roles[1] in output:
            output = output.split(conv.roles[1])[-1].strip()
        return output


def run_inference(num_samples: int = None):
    """Run inference on the MIMICEchoQA dataset."""
    print(f"Loading QA data from {QA_JSON}")
    qa_data = load_qa_data(QA_JSON)
    print(f"Loaded {len(qa_data)} QA samples")

    if num_samples:
        qa_data = qa_data[:num_samples]
        print(f"Processing {num_samples} samples")

    model = LLavaMedInference()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for sample in tqdm(qa_data, desc="Running llava-med inference"):
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
            image = extract_frame_from_dicom(dicom_path, frame_idx=-1)
            question = format_question_with_options(sample)
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

    output_file = OUTPUT_DIR / "llava-med_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")

    correct = sum(1 for r in results if not r["error"] and r["correct_option"] in r["prediction"])
    total = sum(1 for r in results if not r["error"])
    if total > 0:
        print(f"Accuracy: {correct}/{total} = {correct/total:.2%}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run LLaVA-Med inference on MIMICEchoQA")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples to process (default: all)")
    args = parser.parse_args()
    run_inference(args.num_samples)


if __name__ == "__main__":
    main()
