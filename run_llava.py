"""
LLaVA-Med Inference Script
Run visual question answering on echocardiogram images using LLaVA-Med model.
Requires separate conda environment with transformers==4.36.2
"""

import json
import argparse

import torch
from PIL import Image
from tqdm import tqdm

# LLaVA imports
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

from utils import (
    load_qa_data,
    video_path_to_dicom_path,
    extract_frame_from_dicom,
    format_question_with_options,
    save_results,
    OUTPUT_DIR,
    QA_JSON,
)


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
        prompt = f"You are an expert cardiologist analyzing echocardiogram images. {question}\n\nYour response must start with the answer option letter (A, B, C, or D), followed by your reasoning."
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


def run_inference(num_samples: int = None, save_every: int = 50):
    """Run inference on the MIMICEchoQA dataset."""
    print(f"Loading QA data from {QA_JSON}")
    qa_data = load_qa_data()
    print(f"Loaded {len(qa_data)} QA samples")

    if num_samples:
        qa_data = qa_data[:num_samples]
        print(f"Processing {num_samples} samples")

    output_file = OUTPUT_DIR / "llava-med_results.json"

    # Resume from existing results if available
    processed_ids = set()
    results = []
    if output_file.exists():
        with open(output_file, "r") as f:
            results = json.load(f)
            processed_ids = {r["messages_id"] for r in results}
        print(f"Resuming from {len(results)} existing results")

    model = LLavaMedInference()

    for i, sample in enumerate(tqdm(qa_data, desc="Running llava-med inference")):
        if sample["messages_id"] in processed_ids:
            continue
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

            is_correct = prediction.strip()[:1].upper() == sample["correct_option"]
            results.append({
                "messages_id": sample["messages_id"],
                "question": sample["question"],
                "prediction": prediction,
                "ground_truth": sample["answer"],
                "correct_option": sample["correct_option"],
                "structure": sample["structure"],
                "view": sample["view"],
                "error": False,
                "is_correct": is_correct
            })

        except Exception as e:
            print(f"Error processing {sample['messages_id']}: {e}")
            results.append({
                "messages_id": sample["messages_id"],
                "prediction": f"ERROR: {str(e)}",
                "ground_truth": sample["answer"],
                "correct_option": sample["correct_option"],
                "error": True,
            })

        # Periodic save
        if (i + 1) % save_every == 0:
            save_results(results, output_file)

    # Final save
    save_results(results, output_file)
    return results


def main():
    parser = argparse.ArgumentParser(description="Run LLaVA-Med inference on MIMICEchoQA")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples to process (default: all)")
    args = parser.parse_args()
    run_inference(args.num_samples)


if __name__ == "__main__":
    main()
