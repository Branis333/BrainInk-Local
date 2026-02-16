# BrainInk-Local

## Demo Video

[Watch the project demo](https://drive.google.com/file/d/1ZG2nh67_f3T0Or7vT-Qzk8hdMsZoEZAN/view?usp=sharing)

BrainInk-Local is my research workspace for building an AI grading pipeline for handwritten student work.

This repo is not just a single model experiment. It documents a real implementation journey across three model strategies, each chosen to solve the limits of the previous one.

---

## Project Goal

Build a practical system that can:

1. read handwritten student responses from images,
2. reason about correctness using assignment context/rubrics,
3. generate useful feedback for students and teachers,
4. run with realistic compute and memory constraints.

---

## My Implementation Journey (What I actually did)

### 1) First approach: DeepSeek Janus-Pro (single multimodal base model)

I started with **DeepSeek Janus-Pro-1B** as the main multimodal base model.

- Why I chose it:
  - It is a unified multimodal model for image + text understanding/generation.
  - It matched my original idea: one model handling the full grading flow.
- What worked:
  - Strong multimodal behavior and a clean foundation for LoRA-style fine-tuning.
- Why I paused it:
  - The end-to-end pipeline became too heavy for my target constraints.
  - Memory/storage/runtime costs were higher than what I wanted for iterative research.

Result: **promising quality, but too expensive to continue as my primary path right now.**

Related notebook:
- `first.ipynb`

---

### 2) Second approach: split pipeline (FLAN-T5 + OpenAI CLIP)

After pausing Janus, I moved to a two-model design:

- **Vision side:** `openai/clip-vit-base-patch32`
- **Language/reasoning side:** `google/flan-t5-small`

Why I tried this:
- Much smaller memory footprint.
- Easier to run and iterate on local/limited environments.
- Cleaner modularity (vision encoder separate from text generation).

What improved:
- Significant reduction in space and compute required.

Main limitations I hit:
- These models are older compared to newer compact VLM stacks.
- CLIP is excellent for image-text similarity/classification, but it is not a dedicated OCR/handwriting reader.
- For handwritten grading, text extraction quality becomes the bottleneck.

Result: **efficient and lightweight, but weak for reliable handwriting text reading.**

Related notebooks:
- `vision.ipynb`
- `think.ipynb`

---

### 3) Current approach: Smol stack from Hugging Face (best so far)

My current and best-performing direction combines:

- **Vision/document understanding:** `docling-project/SmolDocling-256M-preview`
- **Language reasoning/feedback:** `HuggingFaceTB/SmolLM2-360M`

Why this is currently my best option:
- Keeps the system compact and practical.
- Better document/OCR-style capabilities than the CLIP-only stage.
- Better fit for structured extraction + grading pipeline design.

Current gap:
- Even though this stack is much better, handwriting is still domain-specific.
- The base models are strong on text/doc content, but not fully specialized for messy real classroom handwriting by default.

Next step (active research plan):
- Fine-tune on handwritten educational data.
- Start with my current dataset.
- Expand with additional school data collected during research.

Result: **best tradeoff so far between quality, size, and scalability — now moving into handwriting-focused fine-tuning.**

Related notebooks:
- `smol_docling.ipynb`
- `smol_lm.ipynb`

---

## Current Architecture (active direction)

Handwritten image
→ SmolDocling (extract/structure text + layout signals)
→ intermediate structured summary
→ SmolLM2 (grading reasoning + feedback generation)

This keeps vision parsing and language reasoning separate while staying lightweight.

---

## Repository Map

- `first.ipynb` — Janus-Pro path (initial multimodal fine-tuning experiments)
- `vision.ipynb` — CLIP vision-stage experiments
- `think.ipynb` — FLAN-T5 reasoning/feedback-stage experiments
- `smol_docling.ipynb` — SmolDocling vision/document pipeline
- `smol_lm.ipynb` — SmolLM2 reasoning/feedback pipeline
- `test_base.ipynb` / `test.ipynb` — testing/smoke checks
- `tools/validate_vlm_dataset.py` — dataset validation helper
- `tools/make_vlm_template.py` — template generation helper

---

## Why this README changed

The previous README focused mostly on the original Janus fine-tuning flow.

This updated README reflects the actual evolution of the project:

1. Janus-Pro first,
2. then FLAN + CLIP to reduce compute,
3. now SmolDocling + SmolLM2 as the strongest practical direction,
4. with handwriting-specific fine-tuning as the main research objective.

---

## Research Status

- Janus-Pro implementation: **paused** (resource-heavy for current constraints)
- FLAN + CLIP implementation: **completed prototype** (lightweight but OCR-limited)
- Smol pipeline: **active development**
- Handwriting fine-tuning: **next major milestone**

---

## Near-Term Plan

1. curate and clean handwritten training data,
2. define a strict target output schema for grading,
3. fine-tune the Smol-based pipeline on handwriting samples,
4. evaluate on unseen classroom-style responses,
5. improve evidence-grounded feedback quality and robustness.

---

## Notes

- This repo is research-oriented and iterative by design.
- Notebooks represent different stages of model strategy, not duplicate work.
- The active branch of progress is the Smol-based handwriting grading direction.
