"""
MedGemma Inference Script for MIMICEchoQA.
Run: python main.py --num-samples 10
"""

import argparse

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText

from utils import (
    load_qa_data,
    video_path_to_dicom_path,
    extract_frame_from_dicom,
    format_question_with_options,
    save_results,
    OUTPUT_DIR,
    QA_JSON,
)


class MedGemmaInference:
    """MedGemma model inference class."""

    def __init__(self, model_id: str = "google/medgemma-4b-it"):
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
                "content": [{"type": "text", "text": "You are an expert cardiologist analyzing echocardiogram images. Answer the question based on the echocardiogram image provided. Your response must start with the answer option letter (A, B, C, or D), followed by your reasoning."}]
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


def run_inference(num_samples: int = None):
    """Run MedGemma inference on the MIMICEchoQA dataset."""
    print(f"Loading QA data from {QA_JSON}")
    qa_data = load_qa_data()
    print(f"Loaded {len(qa_data)} QA samples")

    if num_samples:
        qa_data = qa_data[:num_samples]
        print(f"Processing {num_samples} samples")

    model = MedGemmaInference()

    results = []
    for sample in tqdm(qa_data, desc="Running MedGemma inference"):
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

    save_results(results, OUTPUT_DIR / "medgemma_results.json")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run MedGemma inference on MIMICEchoQA")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples to process (default: all)")
    args = parser.parse_args()
    run_inference(args.num_samples)


if __name__ == "__main__":
    main()
