<!--
ROCm Blog draft — fill the [bracketed] placeholders after your MI300X run.
Target: ~2,000–3,000 words, ~10–12 min read. Voice: professional but friendly, reproducibility-first.
ROCm Blogs are open source: https://github.com/ROCm/rocm-blogs — check its CONTRIBUTING.md for the
exact front-matter schema + folder layout before opening a PR.
-->
---
blogpost: true
date: [fill: e.g. 20 Oct 2026]
author: Srikanth Arunachalam
tags: Computer Vision, PyTorch, AI/ML, Object Detection
category: Applications & models
language: English
myst:
  html_meta:
    "description lang=en": "Train and serve YOLOv8 object detection on AMD Instinct MI300X with ROCm — setup, benchmarks, and a fully reproducible repo."
    "keywords": "ROCm, YOLOv8, MI300X, object detection, computer vision, PyTorch, HIP, Ultralytics, gfx942"
    "property=og:locale": "en_US"
---

# Running YOLOv8 object detection on AMD Instinct MI300X with ROCm

*Estimated reading time: ~[fill] min*

## 1. Introduction  — the problem

Object detection — finding and drawing boxes around objects in images and video — powers a huge range of
real products: [fill: e.g. retail shelf monitoring, manufacturing defect inspection, wildlife/fish counting,
workplace safety]. Teams building these systems want to **train and serve detectors cost-effectively**, and
**AMD Instinct GPUs with ROCm are now a first-class option** for exactly that.

In this post we train a **custom YOLOv8 detector on an AMD Instinct MI300X** and show two things:
1. It runs with **essentially no code changes** from the usual NVIDIA/CUDA workflow, and
2. Here are the **real training and inference numbers** on MI300X.

Everything is reproducible from a public repo (linked at the end).

## 2. Hardware and software setup

Reproducibility is the whole point, so here is the exact environment:

| Component | Version / detail |
|---|---|
| GPU | AMD Instinct **MI300X** (`gfx942`) — [fill: 1× / 8×] |
| ROCm | [fill: e.g. 6.x / 7.x] |
| PyTorch | ROCm build (`torch.version.hip` is set) |
| Ultralytics (YOLOv8) | [fill: version] |
| Environment | [fill: AMD Developer Cloud GPU Droplet / Docker image name] |

Confirm the GPU and that PyTorch sees ROCm:

```bash
rocminfo | grep -m1 gfx          # expect gfx942
python -c "import torch; print(torch.__version__, '| HIP:', torch.version.hip, '| GPU:', torch.cuda.is_available())"
pip install ultralytics
```

> **Note:** On a ROCm build of PyTorch, the `torch.cuda.*` API transparently maps to **HIP**, so `device=0`
> targets the MI300X. You keep writing `cuda` in code; it runs on AMD.

## 3. The workflow — custom YOLOv8 training

**Dataset.** [fill: name, e.g. a fish-detection dataset] in YOLO format (images + label `.txt` files) with a
`data.yaml` that lists the classes and train/val paths.

**Model.** Start from pretrained `yolov8n.pt` (swap for `s`/`m`/`l` to trade speed for accuracy).

**Train** — the code is identical to the CUDA flow:

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="data.yaml", epochs=[fill], imgsz=640, batch=[fill], device=0)  # device 0 = MI300X
```

[fill: 2–3 sentences on batch size, augmentation, and how long training took.]

### What changed going from CUDA to ROCm?  (spoiler: almost nothing)

| Layer | CUDA (NVIDIA) | ROCm (AMD) |
|---|---|---|
| GPU runtime | CUDA | **HIP** (drop-in) |
| PyTorch | CUDA build | **ROCm build** (`torch.version.hip` set) |
| Your training code | `device=0` | **identical** |
| Ultralytics / YOLOv8 | works | works (pure PyTorch) |

The only real change was the **environment**, not the code — that is the headline for framework-level models.

## 4. Results

[fill after the run — this section earns the blog its credibility.]

| Metric | Value |
|---|---|
| Epochs | [fill] |
| Training time | [fill] |
| mAP@50 | [fill] |
| mAP@50-95 | [fill] |
| Inference throughput (FPS, imgsz 640) | [fill] |

[CHART: add e.g. throughput vs. batch size as a PNG in `./images/` and reference it below.]
![throughput vs batch size](images/throughput.png)

[SCREENSHOT: a few predicted images with boxes — save as `./images/predictions.png`.]
![sample predictions](images/predictions.png)

> **Be transparent** (a ROCm-blog hallmark): note any caveats — e.g. out-of-memory at large batch sizes,
> warm-up effects, or where numbers are single-run vs. averaged.

## 5. Reproduce it

Everything is in the repo:

```bash
git clone https://github.com/srikantharun/CUDA-ROCm-Migration
cd CUDA-ROCm-Migration && pip install -r requirements.txt
jupyter notebook notebooks/yolov8_custom_training.ipynb   # run the ROCm-readiness cell first
```

Dataset source: [fill: link]. Notebook: `notebooks/yolov8_custom_training.ipynb`.

## 6. Bonus — migrating raw CUDA *kernels* to ROCm

Framework-level models port for free, but the deeper skill is porting **custom CUDA kernels**. The companion
notebook `notebooks/cuda_to_rocm_migration_mi300.ipynb` walks through exactly that — **hipify → `hipcc` build →
validate against a NumPy golden reference** — for a vector-add, SAXPY, a softmax reduction, and an RGB→grayscale
computer-vision kernel. [fill: one sentence on a result, e.g. bit-close match to the golden reference.]

## Conclusion & what's next

YOLOv8 trains and serves on AMD Instinct MI300X with ROCm **essentially unchanged**, and the numbers above show
it is a practical option for production computer vision. Next steps: [fill: larger model, multi-GPU with RCCL,
INT8/quantization, or porting a custom CUDA op].

**Acknowledgements:** AMD AI Developer Program (cloud credits), the Ultralytics team.

---

*Author: **Srikanth Arunachalam** — [github.com/srikantharun](https://github.com/srikantharun) · srikantharun.github.io*
