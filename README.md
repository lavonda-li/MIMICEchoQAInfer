# MIMICEchoQA Inference

Visual Question Answering inference on echocardiogram images using MedGemma and LLaVA-Med models.

## Setup

### 1. Create Conda Environment

```bash
# Create and activate the environment
conda create -n echoqa python=3.10 -y
conda activate echoqa

# Install PyTorch with CUDA support (for T4 GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
pip install transformers>=4.50.0 accelerate pydicom pillow numpy pandas tqdm huggingface_hub bitsandbytes sentencepiece einops

# Install LLaVA-Med (required for LLaVA-Med model)
pip install git+https://github.com/microsoft/LLaVA-Med.git
```

### 2. Hugging Face Login (Required for MedGemma)

MedGemma requires accepting the license on Hugging Face:
1. Visit https://huggingface.co/google/medgemma-4b-it
2. Accept the license agreement
3. Login via CLI:

```bash
huggingface-cli login
```

## Usage

### Run inference with MedGemma (default)
```bash
python main.py --model medgemma
```

### Run inference with LLaVA-Med
```bash
python main.py --model llava-med
```

### Run inference with both models
```bash
python main.py --model both
```

### Test with a subset of samples
```bash
python main.py --model medgemma --num-samples 10
```

## GPU Memory Considerations

Both models are optimized for NVIDIA T4 (16GB VRAM):
- **MedGemma 4B**: Uses bfloat16, fits comfortably on T4
- **LLaVA-Med 7B**: Uses float16 with device_map="auto"

## Data

- **QA Dataset**: `/home/lavondali/MIMICEchoQAInfer/Data/mimic-iv-ext-echoqa/1.0.0/MIMICEchoQA/MIMICEchoQA.json`
- **DICOM Files**: `/home/lavondali/mount-folder/MIMIC-Echo-IV/` (GCP bucket mount)
- **Output**: `./outputs/` directory

The script automatically:
1. Reads the QA JSON dataset (622 samples)
2. Maps video paths to DICOM files in the mounted GCP bucket
3. Extracts the middle frame from each multi-frame DICOM
4. Runs inference with the specified model

## Output

Results are saved as JSON files in the `outputs/` directory:
- `medgemma_results.json`
- `llava-med_results.json`

Each result contains:
- `messages_id`: Sample identifier
- `question`: The question asked
- `prediction`: Model's answer
- `ground_truth`: Correct answer
- `correct_option`: Correct option letter (A/B/C/D)
- `structure`: Cardiac structure being queried
- `view`: Echocardiographic view
