# Vexoo Labs AI Engineer Assignment - Summary Report
**Author**: Candidate

### 1. Document Ingestion & Pyramid Approach (Part 1)
To address failing RAG strategies detailed in the research paper such as chunk isolation and context loss, the ingestion script implements an "Agentic Knowledge Distillation" strategy employing a hierarchical structure:
* **Approach**: Simulated a 2-page sliding window approach with configurable character limits to traverse a document, overlapping content context to avoid cut-off semantics.
* **Pyramid Structure**: For each sliding window, we generated four distinct layers: 
  1. **Raw Text** (the unmodified string window)
  2. **Chunk Summary** (a simulated brief LLM extraction/summary)
  3. **Category/Theme Label** (a classification of the chunk's content)
  4. **Distilled Knowledge** (atomic keywords for rapid lookup/concepts)
* **Retrieval Engine**: Employed `sentence-transformers` for multi-layered retrieval. At inference time, the query embeds and computes cosine similarity against *all* levels of the pyramid. This allows generic, theme-based questions to match against the "Category" layer, while targeted, spear-fishing questions find their matches in the "Raw Text" or "Distilled Knowledge" layers seamlessly.

### 2. GSM8K Training Setup (Part 2)
The finetuning pipeline sets up LLaMA 3.2 1B to learn mathematical reasoning step-by-step using a LoRA PEFT architecture:
* **Dataset Prep**: Sourced `openai/gsm8k` via the Hugging Face `datasets` library. We specifically isolated the 3,000 samples for the training set and 1,000 for evaluation. 
* **LoRA Configuration**: Injected Low-Rank Adapters (`r=16`, `alpha=32`, `dropout=0.05`) directly into the attention mechanism (`q_proj`, `k_proj`, `v_proj`, `o_proj`). This mitigates catastrophic forgetting and significantly decreases VRAM constraints, allowing training on consumer hardware.
* **Training Strategy**: Utilized Hugging Face `trl`'s `SFTTrainer` for autoregressive causal language modeling. Formatted text to the traditional `Question: ... \n\nAnswer: ...` structure. We configured gradient accumulation steps to 4 and employed `fp16` precision to lower VRAM utilization without sacrificing accuracy gradients.
* **Evaluation Framework**: Developed an exact match abstraction which parses out the generated numeric response sequentially succeeding the generic `####` delimiter convention used across GSM8K.

### Key Design Decisions & Assumptions
* **Design Decision**: Elected to "flatten" the Knowledge Pyramid nodes into a unified vector space instead of implementing complex graph traversal logic, prioritizing inference-time lookup velocity O(1) and simplifying the query retrieval logic.
* **Architecture Assumption**: The training script utilizes `meta-llama/Llama-3.2-1B` directly, assuming an authenticated environment (`huggingface-cli login`). The code relies heavily on established abstractions (`peft`, `trl`) reducing explicit, cumbersome backend manipulations and retaining standardized stability.

### Bonus: Reasoning-Aware Adapter Component
**Architecture of a "Plug-and-Play" Context Adapter:**
If deploying reasoning schemas for divergent tasks (Math vs. Legal vs. Coding), I would structure a dynamic **Mixture of Adapters (MoA) / Router Pattern**:
1. **Dynamic Classifier**: A lightweight orchestrator model (e.g. a 120M parameter DistilBERT or simple zero-shot router) sits as a gateway, classifying the intent of the incoming user request.
2. **Adapter Swapping**: Depending on the categorized domain (e.g., `<class: GSM8k_Math>`), the backend LLM dynamically activates and swaps the targeted LoRA weights onto the base model (e.g., utilizing Peft Model `.set_adapter()`).
3. **Triggered Execution**: The base LLM routes the input across the specialized reasoning trajectory without the need to switch entire 1B-7B base parameter memory blocks out of VRAM.
This architecture enables essentially unbounded multi-task competency running in parallel instances, dynamically exchanging memory-cheap "reasoning modules" without exceeding the memory footprint of the underlying unified base model.
