# 🍟➡️📝 FoodExtract-Vision — Fine-Tuned SmolVLM2 for Structured Food/Drink Extraction

Fine-tuning [SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) to turn any image into **structured JSON** describing the food and drink items it contains — built and trained end-to-end on a free Google Colab T4 GPU.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uditjainofficial/smolvlm2-structured-food-extraction/blob/main/notebooks/VLM_Finetuning.ipynb)
[![Model on Hugging Face](https://img.shields.io/badge/🤗%20Model-udit74j%2FFoodExtract--Vision--SmolVLM2--500M-blue)](https://huggingface.co/udit74j/FoodExtract-Vision-SmolVLM2-500M-fine-tune-v1)
[![Demo on Hugging Face Spaces](https://img.shields.io/badge/🤗%20Demo-Live%20Space-yellow)](https://huggingface.co/spaces/udit74j/FoodExtract-Vision-v1)

---

## 📌 Overview

Given **any input image**, the model classifies whether it contains food/drink, generates a short caption, and extracts every visible food and drink item into a clean, structured JSON payload — a task general-purpose VLMs handle inconsistently out of the box, but which a small model can learn reliably with targeted fine-tuning.

**Example output:**

```json
{
  "is_food": 1,
  "image_title": "fried calamari",
  "food_items": ["fried calamari"],
  "drink_items": []
}
```

```json
{
  "is_food": 0,
  "image_title": "",
  "food_items": [],
  "drink_items": []
}
```

This kind of structured extraction is a general pattern applicable well beyond food — the same recipe works for receipts, product photos, menus, or any domain where you want a vision-language model to reliably emit a fixed schema instead of free-form text.

---

## 🚀 Live Demo

Try it here: **[huggingface.co/spaces/udit74j/FoodExtract-Vision-v1](https://huggingface.co/spaces/udit74j/FoodExtract-Vision-v1)**

The demo runs on a free-tier Hugging Face Space, so keep in mind:
- It may take **20–30 seconds to wake up** if it's been idle (cold start).
- Inference is CPU/shared-GPU based, so responses are slower than a dedicated deployment.
- The Space compares the **base model** and the **fine-tuned model** side by side on the same image, so you can see the improvement directly.

---

## 🧠 Model Details

| | |
|---|---|
| **Base model** | [`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) |
| **Fine-tuned model** | [`udit74j/FoodExtract-Vision-SmolVLM2-500M-fine-tune-v1`](https://huggingface.co/udit74j/FoodExtract-Vision-SmolVLM2-500M-fine-tune-v1) |
| **Task** | Image → structured JSON (food/drink classification + item extraction) |
| **Training method** | Full fine-tuning of the language model, **vision encoder frozen** |
| **Framework** | 🤗 Transformers + TRL (`SFTTrainer`) |
| **Hardware used** | Single NVIDIA T4 (Google Colab, free tier) |
| **Precision** | `float16` (T4-safe; swap to `bfloat16` on newer GPUs) |

### Why freeze the vision encoder?

The vision encoder already produces strong general-purpose image representations. Freezing it and only training the language-model half keeps the trainable parameter count low, speeds up training significantly, and avoids catastrophic forgetting of visual features — while still letting the model learn the new structured-output *format* it hasn't seen before.

---

## 📊 Dataset

- **Source:** [`mrdbourke/FoodExtract-1k-Vision`](https://huggingface.co/datasets/mrdbourke/FoodExtract-1k-Vision) (Hugging Face Hub)
- **Split:** 80% train / 20% validation (seeded shuffle for reproducibility)
- **Format:** each sample is an image paired with a target JSON label describing food/drink content

Each sample is converted into a chat-style conversation (`system` → `user` (image + instruction) → `assistant` (target JSON)) before training, matching the format SmolVLM2 expects.

---

## ⚙️ Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 4 |
| Per-device batch size | 1 |
| Gradient accumulation steps | 16 (effective batch size 16) |
| Gradient checkpointing | Enabled |
| Optimizer | `adamw_torch_fused` |
| Learning rate | `2e-4` (constant schedule, 3% warmup) |
| Max grad norm | 1.0 |
| Eval / save strategy | Every epoch (best model kept) |

Full configuration is defined via `SFTConfig` in the notebook — see [`notebooks/VLM_Finetuning.ipynb`](notebooks/VLM_Finetuning.ipynb).

---

## 📁 Repository Structure

```
.
├── notebooks/
│   ├── 01_fine_tuning.ipynb        # Initial fine-tuning experiment
│   └── VLM_Finetuning.ipynb        # Final end-to-end notebook (train → evaluate → deploy)
├── demos/
│   └── FoodExtract-Vision-v1/
│       ├── app.py                  # Gradio app (base vs. fine-tuned comparison)
│       ├── requirements.txt        # Space dependencies
│       └── README.md               # Hugging Face Space card
├── fix_notebook.py                 # Utility: strips broken Colab widget metadata before commits
├── pyproject.toml
└── README.md
```

---

## 🏗️ How It Was Built

The end-to-end pipeline, all in [`VLM_Finetuning.ipynb`](notebooks/VLM_Finetuning.ipynb):

1. **Load & inspect** the `FoodExtract-1k-Vision` dataset
2. **Format** each sample into chat-style `system` / `user` / `assistant` messages
3. **Sanity-check** the base (non-fine-tuned) model's output before training
4. **Freeze** the vision encoder, keeping only the language-model weights trainable
5. **Fine-tune** using TRL's `SFTTrainer` with a custom data collator (image + text batching, loss-masked padding/image tokens)
6. **Evaluate** by plotting train/validation loss and comparing base vs. fine-tuned outputs on held-out samples
7. **Push** the fine-tuned model to the Hugging Face Hub
8. **Build & deploy** a Gradio demo to Hugging Face Spaces

---

## 🖥️ Running Locally

```bash
git clone https://github.com/uditjainofficial/smolvlm2-structured-food-extraction.git
cd smolvlm2-structured-food-extraction

pip install -r demos/FoodExtract-Vision-v1/requirements.txt
```

Run the Gradio demo locally (pulls the fine-tuned model from the Hub):

```bash
cd demos/FoodExtract-Vision-v1
python app.py
```

To reproduce training, open [`notebooks/VLM_Finetuning.ipynb`](notebooks/VLM_Finetuning.ipynb) in Google Colab (badge above) with a GPU runtime (T4 or better).

---

## 🛠️ Tech Stack

- [Transformers](https://github.com/huggingface/transformers) — model + pipeline
- [TRL](https://github.com/huggingface/trl) — `SFTTrainer` for supervised fine-tuning
- [Datasets](https://github.com/huggingface/datasets) — data loading
- [Gradio](https://github.com/gradio-app/gradio) — interactive demo UI
- [Hugging Face Hub](https://huggingface.co/) — model + Space hosting

---

## ⚠️ Limitations

- Fine-tuned on a relatively small dataset (~1k samples) — performance on out-of-distribution images (unusual cuisines, heavily occluded items) may be inconsistent.
- Trained for a fixed output schema; extending the JSON structure requires re-fine-tuning.
- The free-tier Colab T4 setup caps batch size and training time; results may improve with a larger GPU, more epochs, or a larger base model (e.g. SmolVLM2-2.2B).


## 🙌 Acknowledgments

- [HuggingFaceTB](https://huggingface.co/HuggingFaceTB) for the SmolVLM2 model family
- [mrdbourke](https://huggingface.co/mrdbourke) for the `FoodExtract-1k-Vision` dataset and the fine-tuning tutorial this project follows 