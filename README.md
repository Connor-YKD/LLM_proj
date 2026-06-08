This repository contains the code used for the thesis **Semantic Instability and Sequential Score Dynamics under Meaning-Preserving Reformulations in Large Language Models**.

The project studies two complementary uses of meaning-preserving reformulations in large language models:

- **Path 1:** reformulations as a diagnostic of reformulation-sensitive semantic instability
- **Path 2:** reformulations as sequential context for studying score dynamics under structured prompt history

The main experiments were run on the **PopQA-TP** reformulation dataset using **Qwen2.5-7B-Instruct**. 

Detailed implementation choices are documented in the inline comments of the scripts, and the interpretation of the reported quantities should be read together with the thesis.

---

## Environment

Recommended environment:

- Python 3.12
- PyTorch
- transformers
- datasets

---

## Main experimental settings

### Common settings

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Dataset: `ibm-research/popqa-tp`
- Temperature: `0.7`
- Top-p: `0.95`
- Maximum new tokens: `8`

### Path 1

- Number of reformulations per question: `4`
- Repeated batches per reformulation: `4`
- Sampled answers per batch: `4`

### Path 2

- Number of reformulations per question: `4`
- Samples per node: `4`

---

## Path 1: reformulation-sensitive semantic instability

Path 1 compares:

- **within-transformation variability**, based on repeated sampled batches under the same reformulation
- **between-transformation variability**, based on batch-level distributions across different reformulations

The main derived quantity is the excess instability `E_q = B_q - W_q`

Positive values indicate that cross-reformulation discrepancies exceed within-prompt sampling variability on average.

### Path 1 outputs

The Path 1 scripts produce JSON files containing, for each run:

- run-level summary statistics
- timeout skip information
- question-level results
- transformation-level results
- semantic cluster maps
- representative high-instability examples

These outputs are used for the tables and discussion in the Path 1 results section of the thesis.

---

## Path 2: finite-tree score dynamics under sequential reformulation

Path 2 treats reformulations sequentially.

For each latent question:

- the original prompt is used as the root node
- the remaining reformulations are appended in all admissible orders without reuse
- previous question-answer pairs are written back into the prompt history
- the correctness-oriented score is tracked over the resulting finite reformulation tree

### Path 2 outputs

The Path 2 scripts produce JSON files containing, for each run:

- run-level score summaries
- timeout skip information
- question-level results
- node-level scores and histories
- depthwise summaries
- representative examples of stable correct, stable incorrect, and seed-sensitive behaviour

These outputs are used for the tables and discussion in the Path 2 results section of the thesis.

## Reproducing the thesis results

The main reported thesis results correspond to the **stricter filtering setting**.

- Initial strict-loose comparison: selection seed `100`, generation seeds `[4, 16, 64]`
- Main stricter multi-seed evaluation: selection seed `100`, generation seeds `[4, 16, 64, 256, 3, 9, 27, 81]`

