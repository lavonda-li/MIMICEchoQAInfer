"""Shared utilities for MIMICEchoQA inference."""

import json
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image


# Paths
DATA_DIR = Path("/home/lavondali/MIMICEchoQAInfer/Data/mimic-iv-ext-echoqa/1.0.0/MIMICEchoQA")
DICOM_DIR = Path("/home/lavondali/mount-folder/MIMIC-Echo-IV")
QA_JSON = DATA_DIR / "MIMICEchoQA.json"
OUTPUT_DIR = Path("/home/lavondali/MIMICEchoQAInfer/outputs")


def load_qa_data(json_path: Path = QA_JSON) -> list:
    """Load the QA dataset."""
    with open(json_path, "r") as f:
        return json.load(f)


def video_path_to_dicom_path(video_path: str) -> Path:
    """
    Convert video path from JSON to DICOM path.
    Example: mimic-iv-echo/0.1/files/p13/p13265185/s97751020/97751020_0026.mp4
           -> /mount-folder/MIMIC-Echo-IV/p13/p13265185/s97751020/97751020_0026.dcm
    """
    parts = video_path.split("/")
    patient_prefix = parts[3]
    patient_id = parts[4]
    study_id = parts[5]
    filename = parts[6].replace(".mp4", ".dcm")
    return DICOM_DIR / patient_prefix / patient_id / study_id / filename


def extract_frame_from_dicom(dicom_path: Path, frame_idx: int = -1) -> Image.Image:
    """
    Extract a single frame from a DICOM file.
    frame_idx=-1 uses the middle frame.
    """
    dcm = pydicom.dcmread(dicom_path)
    pixel_array = dcm.pixel_array

    if len(pixel_array.shape) >= 3:
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


def save_results(results: list, output_file: Path):
    """Save results to JSON and print accuracy."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")

    correct = sum(1 for r in results if r.get("is_correct"))
    total = sum(1 for r in results if not r.get("error"))
    if total > 0:
        print(f"Accuracy: {correct}/{total} = {correct/total:.2%}")
