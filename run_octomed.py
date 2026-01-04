"""
OctoMed-7B Inference Script for MIMICEchoQA.
Run: python run_octomed.py --num-samples 10
"""

import argparse
import json
import re

import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from utils import (
    load_qa_data,
    video_path_to_dicom_path,
    extract_frame_from_dicom,
    save_results,
    OUTPUT_DIR,
    QA_JSON,
)


class OctoMedInference:
    """OctoMed-7B model inference class."""

    def __init__(self, model_id: str = "OctoMed/OctoMed-7B"):
        print(f"Loading OctoMed model: {model_id}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=262144,
            max_pixels=262144,
        )
        print("OctoMed model loaded successfully")

    def predict(self, image: Image.Image, question: str) -> str:
        """Run inference on a single image with a question."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.6,
                top_p=0.95,
                do_sample=True,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]


def format_question_for_octomed(sample: dict) -> str:
    """Format the question with multiple choice options for OctoMed."""
    question = sample["question"]
    options = []
    for opt in ["A", "B", "C", "D"]:
        opt_text = sample.get(f"option_{opt}", "")
        if opt_text:
            options.append(f"{opt}. {opt_text}")

    formatted = question
    if options:
        formatted += "\n\nOptions:\n" + "\n".join(options)
    formatted += "\n\nPlease reason step-by-step, and put your final answer within \\boxed{}."
    return formatted


def extract_answer_from_response(response: str) -> str:
    """Extract the answer letter from model response."""
    # Try to find answer in \boxed{...}
    matches = re.findall(r'\\boxed\{([^}]+)\}', response)
    if matches:
        answer = matches[-1].strip().upper()
        if answer in ['A', 'B', 'C', 'D']:
            return answer
        # Handle cases like "B. Mild" inside boxed
        if answer and answer[0] in ['A', 'B', 'C', 'D']:
            return answer[0]

    # Fallback: find first A, B, C, or D in response
    for char in response:
        if char.upper() in ['A', 'B', 'C', 'D']:
            return char.upper()
    return ""


def run_inference(num_samples: int = None, save_every: int = 50):
    """Run OctoMed inference on the MIMICEchoQA dataset."""
    print(f"Loading QA data from {QA_JSON}")
    qa_data = load_qa_data()
    print(f"Loaded {len(qa_data)} QA samples")

    if num_samples:
        qa_data = qa_data[:num_samples]
        print(f"Processing {num_samples} samples")

    output_file = OUTPUT_DIR / "octomed_results.json"

    # Resume from existing results if available
    processed_ids = set()
    results = []
    if output_file.exists():
        with open(output_file, "r") as f:
            results = json.load(f)
            processed_ids = {r["messages_id"] for r in results}
        print(f"Resuming from {len(results)} existing results")

    model = OctoMedInference()

    for i, sample in enumerate(tqdm(qa_data, desc="Running OctoMed inference")):
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
            question = format_question_for_octomed(sample)
            prediction = model.predict(image, question)

            extracted_answer = extract_answer_from_response(prediction)
            is_correct = extracted_answer == sample["correct_option"]
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
    parser = argparse.ArgumentParser(description="Run OctoMed inference on MIMICEchoQA")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples to process (default: all)")
    args = parser.parse_args()
    run_inference(args.num_samples)


if __name__ == "__main__":
    main()
