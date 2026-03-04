# BrainInk-Local

## Use BrainInk (Live Website)

Use the platform directly here first: https://brainink.org

## Final version of the product Demo Video

[Watch the project demo](https://drive.google.com/file/d/1pdyj0gMAaNhDGoaF4PEfSIWeAew892a-/view?usp=sharing)

## Initial software product Demo Video

[Watch the project demo](https://drive.google.com/file/d/1ZG2nh67_f3T0Or7vT-Qzk8hdMsZoEZAN/view?usp=sharing)

BrainInk-Local is my AI research and training workspace for handwritten-student grading.

It contains the model implementation history, fine-tuning experiments, evaluation logic, and local AI pipeline experiments that feed into the larger BrainInk system.

---

## Project Goal

Build a practical system that can:

1. read handwritten student responses from images,
2. extract reliable text from noisy classroom handwriting,
3. reason over assignment context/rubrics,
4. generate feedback for students and teachers,
5. run with realistic compute and memory constraints.

---

## Model Implementation and Training (Updated)

### Data preparation and training setup (from `extract/`)

Training data is based on essay-form style samples in `extract/transcriptions.csv` with paired image paths.

The preprocessing flow in `extract/update.ipynb`, `extract/new.ipynb`, and `extract/final_update.ipynb` does the following:

- Parses each transcription and removes metadata/header noise (for example sentence-database prefixes and IDs).
- Extracts the cleaner target text from the essay form content before the `Name:` line.
- Resolves image paths consistently, reusing existing crops when available.
- Builds train/validation splits with fixed seeds for reproducibility.
- Runs LoRA/QLoRA-style fine-tuning experiments with controlled hyperparameter changes.

This made the training pipeline stable enough to compare model behavior across stages rather than relying on one-off runs.

### Stage 1: DeepSeek Janus-Pro-1B fine-tuning

Base model used: `deepseek-ai/Janus-Pro-1B`

I first fine-tuned Janus-Pro-1B on the essay-form derived training targets and ran **four experiments** with different hyperparameter settings (epochs, learning rate, LoRA rank/alpha/dropout, accumulation, etc.).

Outcome:

- I selected the best-performing Janus run by validation/loss behavior,
- deployed and tested it in the workflow,
- but practical results were poor because outputs were too unstable and hallucination-prone for reliable grading.

Conclusion: Janus was valuable for experimentation, but not reliable enough for production handwriting extraction in this use case.

### Stage 2: DeepSeek OCR fine-tuning on the same data

Base model used: `deepseek-ai/DeepSeek-OCR`

After Janus deployment results, I switched to a dedicated OCR-first direction while keeping the same dataset philosophy and training pipeline structure from `extract/`.

Outcome:

- Better OCR specialization,
- lower training loss in my runs,
- more consistent extraction behavior for handwriting-focused tasks.

This became the turning point that changed the system architecture.

### Stage 3: Final workflow decision (OCR + LLM)

Instead of forcing a single multimodal model to do everything, I moved to a two-part pipeline:

1. Fine-tuned OCR model extracts text,
2. LLM performs reasoning/grading/feedback.

Final selected models:

- OCR model: `Branis333/hand_writing_ocr`
- Reasoning LLM: `Qwen/Qwen2.5-7B-Instruct`

This is the implementation direction I chose because it is more controllable and better aligned with grading reliability.

---

## Previous Model Paths (Kept for full project history)

These were real implementation stages and are intentionally retained:

### DeepSeek Janus-Pro

- Model: `deepseek-ai/Janus-Pro-1B`
- Role: end-to-end multimodal baseline and early fine-tuning path.
- Status: completed experimentation, not selected as final deployed OCR path.

### FLAN-T5 + OpenAI CLIP

- Models: `google/flan-t5-small` + `openai/clip-vit-base-patch32`
- Role: lightweight split prototype (vision similarity + language reasoning).
- Status: efficient but OCR quality limitations on handwriting.

### Smol stack from Hugging Face

- Models: `docling-project/SmolDocling-256M-preview` + `HuggingFaceTB/SmolLM2-360M`
- Role: compact structured-extraction + reasoning path.
- Status: active/important research branch for lightweight pipeline design.

---

## Hugging Face Models Used (Exact Names)

- `deepseek-ai/Janus-Pro-1B`
- `deepseek-ai/DeepSeek-OCR`
- `Branis333/hand_writing_ocr`
- `Qwen/Qwen2.5-7B-Instruct`
- `google/flan-t5-small`
- `openai/clip-vit-base-patch32`
- `docling-project/SmolDocling-256M-preview`
- `HuggingFaceTB/SmolLM2-360M`

---

## Repository Arrangement (System-Level)

The full BrainInk system is organized as a multi-repository setup:

1. **AI repository** — model research, training, OCR pipelines, and experiments. 
2. **API backend repository** — API-focused backend services/endpoints integration.
3. **Web frontend repository** — browser UI and client workflows.
4. **App frontend repository** — mobile/app client layer.

In this workspace snapshot, the main mapped repositories are:

- `BrainInk-Local` → AI/training repository (this repo)
- `BrainInk-Backend` → backend service repository
- `BrainInk` → web frontend repository
- `Skana` → App frontend repository

### Run and Access by Repository

All core services are already deployed, so local setup is optional.

## Use BrainInk (Live Website)

Use the platform directly here first: https://brainink.org

#### BrainInk-Local (AI Backend)

- Deployed URL: https://brainink-local.onrender.com
- Repository URL: https://github.com/Branis333/BrainInk-Local (this repo)
- Local run, clone repository(optional):

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

#### BrainInk-Backend (API Backend)

- Deployed URL: https://brainink-backend.onrender.com
- Repository URL: https://github.com/Branis333/BrainInk-Backend
- Local run, clone repository(optional): 

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

#### BrainInk (Web Frontend)

- Deployed URL: https://brainink.org
- Repository URL: https://github.com/Stephen30o0/BrainInk
- Local run, clone repository(optional):

```bash
npm install
npm run dev
```

#### Skana (App Frontend)

- The app is not deployed as a web URL.
- Download build: https://expo.dev/accounts/stephen3000/projects/skana-mobile/builds/096810fd-dc57-447d-b579-5b93124b684a
- Repository URL: https://github.com/Branis333/Skana
- If you do not have Android or run into device issues, run locally (optional):

```bash
npm install
npx expo start --clear
```

### Deployed Access Summary

You can directly use the deployed services here:

- https://brainink.org
- https://brainink-backend.onrender.com
- https://brainink-local.onrender.com

Repository flowchart image:

![System Architecture - Multi-Repository Setup](img/Repository%20Setup.png)

---

## BrainInk-Local Structure (This Repository)

```text
BrainInk-Local/
├── backend/                  # local AI backend endpoints/services
├── experiments/              # experiment outputs and analysis artifacts
├── extract/                  # core training notebooks, OCR training flow, adapters/checkpoints
│   ├── update.ipynb
│   ├── new.ipynb
│   ├── final_update.ipynb
│   └── brainink_janus_hw/    # checkpoints, crops, adapters
├── img/                      # architecture/repository diagrams
│   └── Repository Setup.png
├── ocr/                      # OCR notebooks and datasets used in testing
├── tests/                    # evaluation and validation notebooks/data
├── requirements.txt
└── README.md
```

---

## Research Status

- Janus-Pro training and deployment tests: **completed** (not selected due to hallucination behavior).
- DeepSeek-OCR fine-tuning path: **completed and selected for OCR-first architecture**.
- OCR + LLM integration (`Branis333/hand_writing_ocr` + `Qwen/Qwen2.5-7B-Instruct`): **current implementation direction**.
- Smol stack path: **retained as active complementary research branch**.

---

## Notes

- This repository is intentionally iterative and keeps historical model paths.
- Notebooks represent chronological implementation decisions, not duplicate work.
- The active production-oriented direction is OCR extraction first, then LLM grading/reasoning.
