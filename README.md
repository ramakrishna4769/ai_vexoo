# Vexoo Labs AI Engineer Assignment

This repository contains the implementation of the Vexoo Labs AI Engineer assignment.

## Contents
1. **`part1_ingestion.py`**: A Python script demonstrating a sliding window text ingestion mechanism that builds a multi-layered Knowledge Pyramid and executes semantic search matching against those layers.
2. **`part2_training.py`**: A Hugging Face training script using `trl` setting up a LoRA-based Supervised Fine-Tuning pipeline over the GSM8K dataset aimed at LLaMA 3.2 1B.
3. **`summary_report.md`** & **`summary_report.pdf`**: The summary report detailing the approach, the training setup, and thoughts on the reasoning-aware adapter.

## Setup Instructions

Make sure you have Python 3.9 or higher.

```bash
# 1. Create a virtual environment (optional but recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Running Part 1: Sliding Window & Knowledge Pyramid

Execute the ingestion script directly. It will demonstrate how sample text is paginated into chunks, analyzed through multiple layers of information extraction (Raw, Summary, Theme, Concepts), and finally retrieved via a simple vector similarity search using `sentence-transformers`.

```bash
python part1_ingestion.py
```

## Running Part 2: Train a Reasoning Model on GSM8K using LoRA

> **Important**: `meta-llama/Llama-3.2-1B` in this script targets a gated model repository on Hugging Face. You must possess a Hugging Face account authorized for LLaMA weights.

Before executing, assure your environment is logged in securely:
```bash
huggingface-cli login
```

Then run the script:
```bash
python part2_training.py
```

*Note: For safe code review evaluation, `trainer.train()` inside `part2_training.py` has been explicitly commented out avoiding immediate execution on start. Please uncomment it internally prior to starting the training cycle.*
