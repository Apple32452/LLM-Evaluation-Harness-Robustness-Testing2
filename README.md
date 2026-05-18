# LLM Evaluation Harness & Robustness Testing

A reproducible research-engineering project for evaluating causal language models on multiple-choice reasoning, calibration, prompt-robustness, and failure-mode diagnostics.

This repository is designed to demonstrate practical LLM evaluation infrastructure: log-likelihood scoring, confidence estimation, expected calibration error (ECE), negative log-likelihood (NLL), prompt perturbation testing, category-level breakdowns, JSON experiment artifacts, and failure-clustering utilities.

## Highlights

- Built a lightweight evaluation harness for Hugging Face causal language models.
- Evaluated `Qwen/Qwen2.5-0.5B` on both original and prompt-perturbed multiple-choice datasets.
- Scaled the benchmark from 8 original examples to 208 original examples.
- Generated 1,040 prompt-perturbed examples from the 208-example benchmark.
- Measured accuracy, confidence, ECE, NLL, and category-level performance.
- Added infrastructure for TF-IDF/KMeans failure clustering and LoRA/QLoRA fine-tuning extensions.

## Repository Structure

```text
llm-evaluation-harness-robustness-testing/
├── data/
│   ├── eval_questions.jsonl
│   ├── eval_questions_perturbed.jsonl
│   ├── eval_questions_large.jsonl
│   └── eval_questions_large_perturbed.jsonl
├── results/
│   ├── qwen2_5_0_5b_eval.json
│   ├── qwen2_5_0_5b_perturbed_eval.json
│   ├── qwen2_5_0_5b_large_eval.json
│   └── qwen2_5_0_5b_large_perturbed_eval.json
├── scripts/
│   ├── make_large_eval_dataset.py
│   └── run_lm_eval_example.sh
├── src/
│   ├── evaluate_mcq.py
│   ├── perturb_prompts.py
│   ├── cluster_failures.py
│   └── train_lora_skeleton.py
├── requirements.txt
└── README.md
```

## What This Project Measures

The evaluation harness computes:

- Multiple-choice accuracy
- Average model confidence
- Expected Calibration Error (ECE)
- Negative Log-Likelihood (NLL)
- Category-level performance
- Prompt-perturbation robustness
- JSON experiment records for reproducibility

The main categories in the benchmark are:

- Math
- Machine Learning
- Reasoning
- Systems

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install or upgrade the core model packages:

```bash
pip install --upgrade torch transformers accelerate tokenizers safetensors
```

## Run the Small Qwen Evaluation

Evaluate Qwen on the original 8-example dataset:

```bash
python src/evaluate_mcq.py \
  --model Qwen/Qwen2.5-0.5B \
  --data data/eval_questions.jsonl \
  --out results/qwen2_5_0_5b_eval.json
```

Generate perturbed prompts:

```bash
python src/perturb_prompts.py \
  --data data/eval_questions.jsonl \
  --out data/eval_questions_perturbed.jsonl
```

Evaluate Qwen on the 40-example perturbed dataset:

```bash
python src/evaluate_mcq.py \
  --model Qwen/Qwen2.5-0.5B \
  --data data/eval_questions_perturbed.jsonl \
  --out results/qwen2_5_0_5b_perturbed_eval.json
```

## Small Benchmark Results

| Dataset | Examples | Accuracy | Avg. Confidence | ECE | NLL |
|---|---:|---:|---:|---:|---:|
| Original | 8 | 0.6250 | 0.6342 | 0.2855 | 0.7629 |
| Prompt-Perturbed | 40 | 0.6000 | 0.5988 | 0.2581 | 0.7628 |

### Small Benchmark Category Results

| Dataset | Category | Examples | Accuracy | Avg. Confidence |
|---|---|---:|---:|---:|
| Original | Math | 2 | 0.0000 | 0.3538 |
| Original | Machine Learning | 2 | 1.0000 | 0.8435 |
| Original | Reasoning | 2 | 0.5000 | 0.5038 |
| Original | Systems | 2 | 1.0000 | 0.8356 |
| Prompt-Perturbed | Math | 10 | 0.0000 | 0.3259 |
| Prompt-Perturbed | Machine Learning | 10 | 0.9000 | 0.7843 |
| Prompt-Perturbed | Reasoning | 10 | 0.5000 | 0.4643 |
| Prompt-Perturbed | Systems | 10 | 1.0000 | 0.8208 |

## Build the Larger Dataset

Generate the 208-example larger benchmark:

```bash
python scripts/make_large_eval_dataset.py
```

Confirm the number of examples:

```bash
wc -l data/eval_questions_large.jsonl
```

Expected output:

```text
208 data/eval_questions_large.jsonl
```

Generate the larger prompt-perturbed benchmark:

```bash
python src/perturb_prompts.py \
  --data data/eval_questions_large.jsonl \
  --out data/eval_questions_large_perturbed.jsonl
```

Confirm the number of perturbed examples:

```bash
wc -l data/eval_questions_large_perturbed.jsonl
```

Expected output:

```text
1040 data/eval_questions_large_perturbed.jsonl
```

## Run the Large Qwen Evaluation

Evaluate Qwen on the 208-example dataset:

```bash
python src/evaluate_mcq.py \
  --model Qwen/Qwen2.5-0.5B \
  --data data/eval_questions_large.jsonl \
  --out results/qwen2_5_0_5b_large_eval.json
```

Evaluate Qwen on the 1,040-example prompt-perturbed dataset:

```bash
python src/evaluate_mcq.py \
  --model Qwen/Qwen2.5-0.5B \
  --data data/eval_questions_large_perturbed.jsonl \
  --out results/qwen2_5_0_5b_large_perturbed_eval.json
```

## Large Benchmark Results

| Dataset | Examples | Accuracy | Avg. Confidence | ECE | NLL |
|---|---:|---:|---:|---:|---:|
| Original Large | 208 | 0.7260 | 0.6242 | 0.1620 | 0.6863 |
| Prompt-Perturbed Large | 1,040 | 0.7452 | 0.6098 | 0.1865 | 0.6976 |

### Large Benchmark Category Results

| Dataset | Category | Examples | Accuracy | Avg. Confidence |
|---|---|---:|---:|---:|
| Original Large | Math | 52 | 0.5000 | 0.3580 |
| Original Large | Machine Learning | 52 | 0.7500 | 0.8664 |
| Original Large | Reasoning | 52 | 0.6538 | 0.3429 |
| Original Large | Systems | 52 | 1.0000 | 0.9297 |
| Prompt-Perturbed Large | Math | 260 | 0.5423 | 0.3427 |
| Prompt-Perturbed Large | Machine Learning | 260 | 0.7692 | 0.8515 |
| Prompt-Perturbed Large | Reasoning | 260 | 0.6692 | 0.3472 |
| Prompt-Perturbed Large | Systems | 260 | 1.0000 | 0.8978 |

## Interpretation

The large-scale Qwen evaluation shows that the harness can run end-to-end on a stronger open-source model and a substantially larger prompt set.

On the 208-example original benchmark, `Qwen/Qwen2.5-0.5B` achieved 72.6% accuracy with 0.1620 ECE and 0.6863 NLL. On the 1,040-example prompt-perturbed benchmark, accuracy was 74.5%, with 0.1865 ECE and 0.6976 NLL.

The results show that the evaluation pipeline can measure not only task accuracy, but also confidence behavior, calibration quality, and robustness under prompt variation. Category-level diagnostics also reveal different behavior by task type: systems questions were easiest for the model, while math and reasoning were harder.

## Failure Clustering

To cluster incorrect model outputs:

```bash
python src/cluster_failures.py \
  --eval-json results/qwen2_5_0_5b_large_eval.json \
  --out results/qwen2_5_0_5b_failure_clusters.json
```

Inspect the output:

```bash
cat results/qwen2_5_0_5b_failure_clusters.json
```

## Optional LoRA / QLoRA Fine-Tuning Skeleton

The repository includes a PEFT fine-tuning template:

```bash
python src/train_lora_skeleton.py
```

This script is intended as a starting point for plugging in a real instruction dataset and comparing base versus fine-tuned model behavior.

## Resume-Ready Bullet

```latex
\resumeItem{Evaluated \texttt{Qwen2.5-0.5B} on \textbf{208 original} and \textbf{1,040 prompt-perturbed} examples across math, ML, reasoning, and systems tasks, measuring accuracy, confidence, ECE, NLL, category diagnostics, and robustness under prompt variation.}
```

```latex
\resumeItem{Measured \textbf{72.6\% accuracy} on the original benchmark and \textbf{74.5\% accuracy} on the prompt-perturbed benchmark, with ECE shifting from \textbf{0.1620} to \textbf{0.1865} and NLL from \textbf{0.6863} to \textbf{0.6976}.}
```

## Notes

The benchmark is intentionally compact and reproducible. The current dataset is useful for demonstrating evaluation infrastructure, prompt perturbation, and diagnostic reporting. Future extensions could include larger benchmark datasets, additional model families, pairwise preference evaluation, stronger failure clustering, and comparison between base and fine-tuned models.
