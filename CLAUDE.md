# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/claude-code) when working with code in this repository.

## Project Overview

MIMICEchoQA Inference - A Python tool for running visual question answering (VQA) inference on echocardiogram images from the MIMIC-IV-Echo dataset using medical vision-language models.

## Architecture

- **main.py**: Single-file inference script containing:
  - `MedGemmaInference`: Wrapper for Google's MedGemma-4B model
  - `LLavaMedInference`: Wrapper for Microsoft's LLaVA-Med model
  - DICOM frame extraction utilities
  - QA dataset loading and processing

## Data Paths

- QA Dataset: `Data/mimic-iv-ext-echoqa/1.0.0/MIMICEchoQA/MIMICEchoQA.json`
- DICOM files: External mount at `/home/lavondali/mount-folder/MIMIC-Echo-IV/`
- Output: `outputs/` directory

## Common Commands

```bash
# Run inference with MedGemma (default)
python main.py

# Run inference with LLaVA-Med
python main.py --model llava-med

# Run both models
python main.py --model both

# Limit to N samples for testing
python main.py --num-samples 100
```

## Dependencies

Key packages (not yet in pyproject.toml):
- torch
- transformers
- pydicom
- pandas
- numpy
- Pillow
- tqdm
- llava (for LLaVA-Med model)

## Key Concepts

- **DICOM Processing**: Echocardiograms are stored as multi-frame DICOM files. The code extracts the middle frame by default for single-frame inference.
- **Multiple Choice QA**: Questions have options A-D, with answers matched against model predictions.
- **Video Path Mapping**: JSON references video paths that are converted to DICOM paths in the mounted filesystem.
