# Surgical-Copilot: Real-Time Blood Segmentation and Hemorrhage Source Tracking in Minimally Invasive Surgery

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Framework](https://img.shields.io/badge/framework-PyTorch%20%7C%20MONAI-orange.svg)](https://monai.io/)
[![Optimization](https://img.shields.io/badge/optimization-NVIDIA%20TensorRT-green.svg)](https://developer.nvidia.com/tensorrt)

> **Course Project — Computer Vision**
> **Department of Engineering "Enzo Ferrari" (DIEF)**
> **University of Modena and Reggio Emilia (UNIMORE)**

## Authors

* [Auri Giulia](https://github.com/GiuliaAuri)
* [Mininno Claudio](https://github.com/minicla03)

---

## 🤖 Can AI Become a Surgical Co-Pilot?
Imagine a robotic assistant that doesn't just follow a surgeon’s lead but acts as a vigilant observer, identifying critical events before they escalate. In Minimally Invasive Surgery (MIS), uncontrolled bleeding is a high-stakes complication. Detecting it is one thing; localizing its exact source amidst smoke, reflections, and complex anatomy is the true frontier of computer vision.

This project challenges the boundaries of simple classification. **Surgical-Copilot** is a dual-stage system capable of synergistic detection: segmenting blood pooling regions while simultaneously pinpointing the arterial point of origin in real-time. Developed as part of the **TRAMIS** (Trustworthy Robotic Assistant for Improved Minimally Invasive Surgery) project, this work provides actionable, high-fidelity visual perception to surgical teams.

---

## 🎯 Core Objectives

* **Dual-Stage Localization:** A unified pipeline handling both region segmentation (*where is the blood?*) and source tracking (*where is it coming from?*).
* **Real-Time Optimization:** Achieving a surgical frame rate of **&ge; 30 FPS** utilizing NVIDIA TensorRT for low-latency inference.
* **Benchmarking with HemoSet:** Training and validating models using the first specialized dataset for hemostasis management automation, employing a strict Patient-Level Group K-Fold cross-validation to prevent data leakage.

---

## The Perturbation Engine: Stress-Testing for Trustworthy AI

A model's performance on clean validation data rarely reflects its reliability in the operating room. To rigorously verify the robustness of our architectures, we engineered a custom **Perturbation Engine** (`PerturbationPipelines`). 

During inference, this engine subjects the models to iterative stress tests, simulating severe real-world endoscopic disruptions:
* **`smoke_only`**: Injects a custom procedural noise pattern (Perlin/Simplex) with alpha blending to accurately mimic cauterization smoke.
* **`specular_only`**: Generates high-intensity saturated blobs to simulate coaxial light reflections on wet tissues.
* **`blur_only`**: Induces severe motion blur caused by rapid instrument handling.
* **`chirurgical_worst_case`**: A simultaneous application of all disruptions to identify the absolute breaking point of the architectures.

---

## Verifying Robustness: The Shift to Temporal Approaches

Our initial benchmarking explored a wide taxonomy of **Spatial Baselines** (U-Net, U-Net++, Swin-UNETR, YOLOv8-Seg). While these models achieved competitive baseline Dice scores, the Perturbation Engine revealed a critical vulnerability. Under scenarios like `blur_only` or `chirurgical_worst_case`, purely spatial models catastrophically failed, with their Dice scores dropping to near zero. 

To effectively verify and achieve clinical robustness, we pivoted to **Temporal/Recurrent Approaches**. By exploiting the chronological sequence of the endoscopic video, we aim to maintain anatomical consistency even when individual frames are heavily degraded:
* **Recurrent Architectures:** Integration of pre-trained spatial encoders (e.g., ResNet18) with recurrent bottleneck modules (**ConvGRU** or **ConvLSTM**).
* **Temporal Dataloading:** Unlike spatial models that require Independent and Identically Distributed (I.I.D.) shuffling, our temporal dataloaders strictly preserve the endoscope's recording sequence, allowing the network to learn the fluid dynamics of a hemorrhage.

This evolution from static image segmentation to dynamic video analysis is the cornerstone of building a truly resilient surgical co-pilot.

---

# 🛠️ Installation

The project requires **Python 3.11** and an NVIDIA GPU supporting CUDA and TensorRT.

```bash
# Clone the repository
git clone https://github.com/minicla03/Surgical-Copilot.git

cd Surgical-Copilot

# Install dependencies
pip install -r requirements.txt

# Run an experiments
python ./Surgical-Copilot/src/surgical_copilot/bench/runner/experiment.py model_key=$CURRENT_MODEL
```
