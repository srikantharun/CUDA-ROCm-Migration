# CUDA-ROCm-Migration

A hands-on, notebook-driven lab for moving GPU work from **CUDA (NVIDIA)** to **AMD ROCm / HIP**,
designed to run on an **AMD Instinct MI300X (gfx942)** instance (e.g. AMD Developer Cloud).

## Two tracks
This repo tells a two-part story — both run on the same MI300X box:

| Track | Notebook | What you learn |
|---|---|---|
| **1 · Kernel migration** (the skill) | `notebooks/cuda_to_rocm_migration_mi300.ipynb` | Hand-port raw **CUDA kernels → HIP** (vector-add, saxpy, softmax, a CV grayscale kernel), build with `hipcc`, validate each against a NumPy **golden reference**. |
| **2 · Real model on AMD** (the proof) | `notebooks/yolov8_custom_training.ipynb` | Train **YOLOv8** on AMD **unchanged** — pure PyTorch runs on ROCm as-is (`device=0`); the only change is the environment. |

### Track 1 — you will:
1. Check the ROCm environment (`hipcc`, `rocminfo`, `rocm-smi`).
2. Write and run your first **HIP** kernel (vector add).
3. Take real **CUDA** sources (`saxpy`, row-`softmax`, `grayscale`) and **hipify** them to HIP.
4. Build with `hipcc` and **validate bit-close against a NumPy golden reference**.
5. (Optional) Migrate a **PyTorch custom CUDA extension** via auto-hipify.
6. Learn the **gotchas** (wavefront-64, warp-shuffle, PTX, atomics).

## Prerequisites (on the MI300 box)
- A ROCm install (6.x+) with `hipcc`, `hipify-perl`, `rocminfo`, `rocm-smi` on PATH
  (they live under `/opt/rocm/bin`). On an AMD cloud image these are usually preinstalled.
- Python 3 with `pip install -r requirements.txt`.

## Get it onto the MI300X instance
```bash
git clone https://github.com/srikantharun/CUDA-ROCm-Migration
cd CUDA-ROCm-Migration
```

## Run the labs
```bash
pip install -r requirements.txt
rocminfo | grep -m1 gfx            # confirm the GPU (expect gfx942 on MI300X)

# Track 1 - kernel migration:
jupyter notebook notebooks/cuda_to_rocm_migration_mi300.ipynb

# Track 2 - YOLOv8 training on AMD (run the ROCm-readiness cell first):
jupyter notebook notebooks/yolov8_custom_training.ipynb
```

## Repo layout
```
notebooks/cuda_to_rocm_migration_mi300.ipynb      # Track 1: step-by-step kernel migration lab
notebooks/yolov8_custom_training.ipynb            # Track 2: YOLOv8 training on AMD (ROCm-ready)
cuda/saxpy.cu, cuda/softmax.cu, cuda/grayscale.cu # CUDA sources to migrate (incl. a CV kernel)
scripts/hipify.sh                                 # convenience hipify + diff
validate.py                                       # golden-reference checks (also used by CI)
.github/workflows/rocm-ci.yml                     # CI stub for a self-hosted AMD runner
```

## Next step after this lab
Hand-migrate `tspeterkim/flash-attention-minimal` (raw CUDA, no official ROCm port) and
validate it against PyTorch `scaled_dot_product_attention` as the golden reference.
