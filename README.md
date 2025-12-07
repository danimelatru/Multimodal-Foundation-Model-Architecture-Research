# Multimodal Foundation Model Architecture Research

### CentraleSupélec | Sep 2025 – Apr 2026
**Supervised by Dr. Zhiguo Zeng**

---

## Project Highlights
* **Novel Architecture Design:** Designing a transformer-based multimodal architecture for Text, Audio, and Time-series data, directly applicable to **robotic perception and state estimation**.
* **Self-Supervised Learning:** Implementing Self-Supervised Learning objectives, specifically Masked Signal Modeling, to pre-train foundation models from scratch on large-scale datasets.
* **Comparative Analysis:** Conducting comparative analysis of Parameter-Efficient Fine-Tuning (PEFT) vs. full pre-training to determine optimal transferability for few-shot generalization.
* **HPC Optimization:** Managing large-scale distributed training jobs on HPC infrastructure, utilizing CUDA optimization for multi-GPU throughput.

---

## Project Structure
This repository contains the implementation and research scripts:

* `momentfm/`: Core architecture and model definitions (Foundation Model).
* `scripts/`: Custom training and testing scripts developed for the CWRU dataset.
* `data/`: Dataset configurations and data loading utilities.
* `checkpoints/`: *[Local Only]* Directory where model weights and training states are saved during execution.