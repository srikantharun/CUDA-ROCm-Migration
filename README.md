# rocm-migration-lab

A hands-on, notebook-driven lab for migrating **CUDA** GPU code to **AMD ROCm / HIP**,
designed to run on an **MI300 (gfx942)** instance.

You will:
1. Check the ROCm environment (`hipcc`, `rocminfo`, `rocm-smi`).
2. Write and run your first **HIP** kernel (vector add).
3. Take real **CUDA** sources (`saxpy`, row-`softmax`) and **hipify** them to HIP.
4. Build with `hipcc` and **validate token-/bit-close against a NumPy golden reference**.
5. (Optional) Migrate a **PyTorch custom CUDA extension** via auto-hipify.
6. Learn the **gotchas** (wavefront-64, warp-shuffle, PTX, atomics).

## Prerequisites (on the MI300 box)
- A ROCm install (6.x+) with `hipcc`, `hipify-perl`, `rocminfo`, `rocm-smi` on PATH
  (they live under `/opt/rocm/bin`). On an AMD cloud image these are usually preinstalled.
- Python 3 with `pip install -r requirements.txt`.

## Get it onto the MI300 instance
```bash
# option A: push this repo to GitHub from your laptop, then on the box:
git clone <your-repo-url> && cd rocm-migration-lab
# option B: copy directly
scp -r rocm-migration-lab user@mi300-host:~/
```

## Run the lab
```bash
pip install -r requirements.txt
jupyter notebook notebooks/cuda_to_rocm_migration_mi300.ipynb
# then run the cells top to bottom
```

## Repo layout
```
notebooks/cuda_to_rocm_migration_mi300.ipynb   # the step-by-step lab
cuda/saxpy.cu, cuda/softmax.cu, cuda/grayscale.cu  # CUDA sources to migrate (incl. a CV kernel)
scripts/hipify.sh                               # convenience hipify + diff
validate.py                                     # golden-reference checks (also used by CI)
.github/workflows/rocm-ci.yml                   # CI stub for a self-hosted AMD runner
```

## Next step after this lab
Hand-migrate `tspeterkim/flash-attention-minimal` (raw CUDA, no official ROCm port) and
validate it against PyTorch `scaled_dot_product_attention` as the golden reference.
