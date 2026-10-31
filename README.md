# Vartan

Exploratory notebook work on small, domain-specific AI models — generative image models and language models built and trained largely from scratch, with an emphasis on Indian languages, Indian visual culture, and agriculture.

This repository is a research log, not a packaged library. The three notebooks are self-contained Colab experiments: each one builds a model, trains it for a short run, and inspects the output. Expect hardcoded Colab paths (`/content/...`), `!pip install` cells, and Google Drive mounts.

## Notebooks

### `Vartan_1.ipynb` — BharatUNet (text-to-image, from scratch)

A from-scratch text-to-image pipeline aimed at Indian subjects and prompts.

- **`SovereignVAE`** — a convolutional VAE with residual blocks, GroupNorm and gradient checkpointing, compressing 512×512 RGB images into a 32-channel latent.
- **`IndicTokenizer`** — a tokenizer built from Unicode ranges, covering Devanagari, Tamil, Bengali, Telugu, Kannada, Malayalam, Gujarati and Punjabi without an external tokenizer dependency. Paired with a 768-dim `IndicEmbedding` ("Swaraj-Encoder").
- **`BharatUNetDiT`** — a diffusion transformer trained with **flow matching** rather than classic DDPM noise prediction; predicts a velocity field conditioned on text and timestep.
- **Training** — `FlowMatchingLoss`, a perceptual loss on VGG16 features for the VAE, separate AdamW optimisers for the DiT and text encoder, cosine LR schedules, and a synthetic Indian-palette dataset so the loop can be validated before real data arrives.
- **Data** — a JSON caption manifest keyed to image folders (`1_saree_fabric`, `2_saree_worn`, `3_lehenga`, `4_sherwani_kurta`, `Punjabi_Pagdi_Fusion`) and a `BharatUNetRealDataset` for real images with captions.
- Ends with a generation grid and an explicit "honest performance audit" of VAE reconstruction quality and sample outputs.

### `agri_slm.ipynb` — Agri-SLM (domain language model)

A small language model fine-tuned for agricultural advice in a Colab-hosted project folder (`agri-slm`).

- Preprocessing and tokenisation with the `microsoft/phi-1_5` tokenizer.
- LoRA fine-tuning on top of `microsoft/phi-1_5`, with checkpoints persisted to Google Drive.
- A standalone `inference.py` that loads a trained or LoRA-merged checkpoint and answers a single query.
- Example queries are soil-and-climate crop recommendations, e.g. N/P/K levels, pH, temperature, humidity and rainfall in, crop suggestion out.
- Includes an evaluation script (`scripts/evaluate.py`) and publishing to the Hugging Face Hub with a generated model card.

### `small_llm.ipynb` — ERD (a small LLM, then distillation)

A compact decoder-only language model written from scratch, used as a distillation student.

- **`ERDConfig` / `ERDModel`** — configurable decoder with `RMSNorm`, **MultiHeadLatentAttention** (latent KV cache), `SwiGLUFFN`, and a pre-norm block structure. Reports both total and active parameters per forward pass.
- **Distillation** — a teacher (a Qwen2.5-0.5B model with LoRA) distils into the from-scratch student on GSM8K, with a temperature-scaled KD loss blended with ground-truth cross-entropy (`alpha` weighting).
- A `StopOnAnswerClose` stopping criterion halts generation when `</answer>` appears, robust to tokenisation.
- Saves the teacher LoRA weights and the student checkpoint separately at the end of the run.

## Running the notebooks

There is no build step — open a notebook in Google Colab (a GPU runtime is assumed) and run the cells top to bottom.

```bash
pip install torch torchvision einops matplotlib pillow requests
pip install transformers datasets accelerate peft sentencepiece
```

Notes:

- `Vartan_1.ipynb` and `small_llm.ipynb` are self-contained and generate their own synthetic or downloaded data.
- `agri_slm.ipynb` expects an `agri-slm.zip` project upload containing `data/preprocess.py`, `train.py`, `scripts/evaluate.py` and a `requirements.txt`.
- Checkpoints are written to `/content/bharatunet/checkpoints`, `./checkpoints/...` and Google Drive, so paths need adjusting outside Colab.

## Status

Prototype / research stage. Training runs are short and intended to prove the pipelines work end to end rather than to produce competitive models. Several cells are exploratory, and the notebooks are not yet consolidated into a reusable package.

## Topics

`ai` · `llm` · `language-model` · `deep-learning` · `text-to-image` · `flow-matching` · `distillation` · `agriculture-ai` · `indian-languages`
