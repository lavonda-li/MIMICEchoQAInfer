# MIMICEchoQA

**MIMICEchoQA** is a benchmark dataset for echocardiogram-based question answering, built from the publicly available MIMIC-IV-ECHO database. It is designed to support the development and evaluation of medical vision-language models (VLMs) that integrate echocardiographic video with diagnostic language understanding.

---

## Abstract

We present **MIMICEchoQA**, a benchmark dataset for echocardiogram-based question answering, built from the publicly available MIMIC-IV-ECHO database. Each echocardiographic study was paired with the closest discharge summary within a 7-day window, and the transthoracic echocardiography (TTE) and ECHO sections were extracted to serve as proxies for cardiologist-authored diagnostic reports. DICOM videos were converted to `.mp4` format, and a language model was used to generate multi-turn, closed-ended Q/A pairs grounded in these reports. To ensure anatomical consistency, a view classification model was used to label each video by its echocardiographic view (e.g., A4C, A3C), enabling the filtering of questions referencing structures not visible in the corresponding video. All generated Q/A pairs were manually reviewed by two board-certified cardiologists to ensure clinical validity. The final dataset consists of 622 high-quality, view-consistent Q/A pairs aligned with real-world echocardiogram videos, offering a valuable resource for developing and evaluating models in echocardiographic visual question answering.

---

## Background

Echocardiography is a widely used, non-invasive imaging modality for assessing cardiac structure, function, and hemodynamics [1]. It plays a central role in diagnosing cardiovascular diseases due to its accessibility and ability to provide real-time insights into heart function [2]. As echocardiographic data continues to accumulate in large hospital systems, there is growing interest in developing machine learning (ML) models to assist clinicians with interpretation. Most existing work has focused on traditional supervised tasks such as view classification, structure segmentation, or function quantification [3,4,5]. While valuable, these tasks often fall short of capturing the diverse clinically relevant queries that physicians routinely ask during patient care.

To bridge this gap, we introduce **MIMICEchoQA**, a clinician-validated question-answering benchmark constructed from the MIMIC-IV-ECHO dataset [6]. MIMICEchoQA focuses on core cardiology topics such as ejection fraction, valvular abnormalities, chamber size, and pericardial effusion—questions that reflect the actual diagnostic priorities of practicing cardiologists. Each question-answer pair is grounded in both echocardiogram videos and their corresponding diagnostic reports, and the dataset has been reviewed by clinical experts for medical validity and consistency.

Unlike prior benchmarks, MIMICEchoQA is specifically designed to evaluate the capabilities of medical vision-language models (VLMs)—models that integrate visual input (e.g., echo video) and textual data (e.g., reports, prompts) to perform clinically meaningful tasks. By providing a standardized, expert-reviewed set of echocardiogram-grounded Q/A examples, MIMICEchoQA aims to facilitate progress in multimodal AI systems for real-world cardiology applications.

---

## Methods

MIMICEchoQA was constructed from the **MIMIC-IV-ECHO** dataset [6], which includes echocardiogram videos and metadata collected from patients at Beth Israel Deaconess Medical Center. Each echocardiographic study was linked to the nearest discharge summary from **MIMIC-IV-Note** [7] within a ±7 day window. We retained only those discharge summaries that contained echo-specific sections or keywords such as “ECHO” or “TTE”, resulting in a pool of **1,200 unique patients** with corresponding echocardiogram studies and relevant diagnostic reports.

From each selected report, we extracted the echo-specific section text and used a securely hosted **Qwen-2-72B-Instruct** large language model (LLM) to generate multiple candidate question-answer (Q/A) pairs. The questions were designed to be closed-ended, clinically grounded, and aligned with core echocardiographic concepts such as ejection fraction, valvular severity, chamber size, and pericardial effusion.

To ensure baseline quality, we applied a first-pass filtering step using the same LLM to eliminate noisy or underspecified questions. Specifically, we removed Q/A pairs that lacked clinical specificity, referenced ambiguous findings, or contained unclear answer formats. Multiple-choice answers were standardized using consistent clinical labels (e.g., *Normal, Mild, Moderate, Severe*).

From this filtered pool, we randomly sampled **1,000 unique video–question pairs** and evenly divided them between two board-certified cardiologists. Each cardiologist independently reviewed **500 unique samples**, evaluating each pair across three dimensions:
- Whether the question is clinically relevant and grounded in the diagnostic report.
- Whether the answer is correct based on the provided text.
- Whether the question is visually answerable from the assigned echocardiographic video.

After manual review, **622 video–question pairs** met all criteria and were retained in the final benchmark. Each example corresponds to a unique echocardiogram video from a different patient. Since each reviewer evaluated a disjoint subset of examples, there were no overlapping reviews or inter-reviewer inconsistencies to resolve.

---

## Data Description

Each entry in **MIMICEchoQA** includes the following components:

- **Video**: A `.mp4` transthoracic echocardiogram clip derived from original DICOM files. Each is labeled by its echocardiographic view (e.g., A4C, A3C).
- **Question and Answer**: A clinically relevant, closed-ended multiple-choice question with four answer options (A–D), grounded in the associated report and video. The correct answer is explicitly marked.
- **Anatomical Structure**: The cardiac structure referenced in the question (e.g., *Mitral Valve, Left Ventricle*), supporting structured and view-aware reasoning.
- **Report Context**: The relevant sentence from the clinical report used to justify the answer, along with the full **impression** section.
- **Metadata**: Study ID, video filename, view label, and test split assignment.

---

## Dataset Statistics

- **Total entries**: 622  
- **Unique echo videos**: 622  
- **Unique patients**: 622  
- **Questions per video**: 1  
- **Unique echocardiographic views**: 48  

### Most common views:
- A4C: 72  
- Subcostal 4C: 52  
- A3C: 51  
- PLAX Zoom Out: 49  
- PLAX: 38  
- Doppler A5C: 33  
- PSAX (apex): 30  
- PSAX (papillary muscles): 27  
- A2C: 26  

### Most common cardiac structures:
- Left Ventricle: 162  
- Aortic Valve: 132  
- Mitral Valve: 87  
- Pericardium: 82  
- Left Atrium: 38  
- Atrial Septum: 24  
- Right Ventricle: 23  
- Tricuspid Valve: 23  
- Aorta: 18  
- Pulmonary Artery: 12  
- Right Atrium: 11  
- Segmental Wall Motion: 5  
- Pulmonic Valve: 4  
- Inferior Vena Cava (IVC): 1  

This diversity in views and anatomical focus allows MIMICEchoQA to test a model’s ability to integrate **view-specific visual reasoning** with **clinically grounded language understanding**.

---

## Example Entry

```json
{
  "messages_id": "5921e583-9df1-45d6-825f-4b3399b0f24a",
  "videos": ["mimic-iv-echo/0.1/files/p10/p10119872/s98097718/98097718_0047.mp4"],
  "question": "What is the severity of mitral stenosis?",
  "answer": "Normal",
  "correct_option": "A",
  "option_A": "Normal",
  "option_B": "Mild",
  "option_C": "Moderate",
  "option_D": "Severe",
  "study": "s98097718",
  "image": "98097718_0047",
  "structure": "Mitral Valve",
  "report": "impression: suboptimal image quality. small...",
  "exact_sentence": "Severe mitral annular calcification with trivial stenosis.",
  "view": "A3C",
  "split": "test"
}
```

# Ethics Statement

This project builds upon the publicly available **MIMIC-IV-ECHO** and **MIMIC-IV-Note** datasets [6,7], which are de-identified and distributed under strict data use agreements. All patient data used in this project were previously approved for credentialed distribution by the institutional review board (IRB) at Beth Israel Deaconess Medical Center and the Massachusetts Institute of Technology.

No additional patient data were collected. All derived examples and outputs are fully de-identified and consistent with the original de-identification protocols.

Question-answer pairs were generated using a securely hosted instance of the **Qwen-2-72B-Instruct** large language model, deployed in an access-controlled research environment. No patient data were exposed to public APIs or third-party LLMs during any stage of development.

This work complies with PhysioNet’s ethical data sharing standards and was conducted entirely by credentialed researchers under approved data use agreements.

# References

1. Otto CM. Textbook of clinical echocardiography. Elsevier Health Sciences; 2013 Apr 25.

2. Madani A, Arnaout R, Mofrad M, Arnaout R. Fast and accurate view classification of echocardiograms using deep learning. NPJ digital medicine. 2018 Mar 21;1(1):6.

3. Mortada MJ, Tomassini S, Anbar H, Morettini M, Burattini L, Sbrollini A. Segmentation of anatomical structures of the left heart from echocardiographic images using deep learning. Diagnostics. 2023 May 9;13(10):1683.

4. Zhang J, Gajjala S, Agrawal P, Tison GH, Hallock LA, Beussink-Nelson L, Lassen MH, Fan E, Aras MA, Jordan C, Fleischmann KE. Fully automated echocardiogram interpretation in clinical practice: feasibility and diagnostic accuracy. Circulation. 2018 Oct 16;138(16):1623-35.

5. Ouyang D, He B, Ghorbani A, Yuan N, Ebinger J, Langlotz CP, Heidenreich PA, Harrington RA, Liang DH, Ashley EA, Zou JY. Video-based AI for beat-to-beat assessment of cardiac function. Nature. 2020 Apr;580(7802):252-6.

6. Gow, B., Pollard, T., Greenbaum, N., Moody, B., Johnson, A., Herbst, E., Waks, J. W., Eslami, P., Chaudhari, A., Carbonati, T., Berkowitz, S., Mark, R., & Horng, S. (2023). MIMIC-IV-ECHO: Echocardiogram Matched Subset (version 0.1). PhysioNet. https://doi.org/10.13026/ef48-v217.

7. Johnson, A., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023). MIMIC-IV-Note: Deidentified free-text clinical notes (version 2.2). PhysioNet. https://doi.org/10.13026/1n74-ne17.

8. Zou Group. OpenBiomedVid [Internet]. GitHub; 2025. Available from: https://github.com/zou-group/OpenBiomedVid [Accessed 3 Oct 2025].