# NEXT STEPS — ROCm / Hipbridge / CV contribution

_Last updated: 2026-10-11_

## Where things stand
- **Cloud credits:** requested from the AMD AI Developer Program (~3-day wait, expected ~Mon 13 Oct).
  Console: `devcloud.amd.com` → **GPU Droplets**. Billing currently shows **$0 credits applied** — must be active before launching (else it bills your own card; MI300X ≈ a few $/hr).
- **Repos:**
  - **`CUDA-ROCm-Migration`** (this repo — portfolio hub): kernel-migration lab + YOLOv8 notebook, blog draft `blog/yolov8-rocm-mi300x/`, CV grayscale CUDA kernel.
  - **`hipbridge` fork** — `github.com/srikantharun/hipbridge`, branch **`cv-kernels`**: scaffolded RGB→grayscale (CUDA ref + tuned HIP + Triton `cv.py` + `docs/cv-contribution-plan.md`). **Not yet wired into a Suite; no PR yet.**
- **Hipbridge thread:** reply **drafted** (offering CV coverage + the operands design question). Waiting to post / for maintainer response.

## TODO when the thread reply comes back
- If maintainer **welcomes CV** + answers the operands question (`Suite.extras` vs a `shapes.image()` extension):
  1. Wire `RGB_TO_GRAY` into `verify/suites.py` (proposed literal is in `docs/cv-contribution-plan.md`).
  2. Add `shapes.image()` if that's the chosen path.
- If they push back, adjust approach per their guidance before any PR.

## TODO when cloud credits arrive (steps 2 & 3)
**Droplet setup**
- Launch **MI300X (gfx942)** GPU Droplet, **ROCm + PyTorch** image.
- **Add your SSH key** (same one as GitHub) so Claude can SSH-drive the terminal work.
- Sanity: `rocminfo | grep gfx942` · `python -c "import torch; print(torch.version.hip)"`

**Step 2 — Hipbridge verify (drive via SSH terminal, NOT notebook)**
- `git clone git@github.com:srikantharun/hipbridge && cd hipbridge && git checkout cv-kernels`
- `pip install -e '.[kernels]'`  (needs Triton Linux wheel + ROCm)
- Run `hipbridge.verify` for `rgb_to_gray` across **fp32/fp16/bf16** vs the float64 oracle.
- Capture evidence to `results/`, then wire the `Suite` and re-run.

**Step 3 — PR**
- Open PR `srikantharun:cv-kernels` → `Core-Creates:main` with `results/` evidence.

**YOLOv8 blog (drive via Jupyter)**
- Run `notebooks/yolov8_custom_training.ipynb` (ROCm-readiness cell first).
- Quick validation (pretrained inference + 3-epoch smoke) → then optional fuller train.
- Showcase **CPX compute partitioning**: `sudo amd-smi set --gpu all --compute-partition CPX` → parallel YOLOv8 inference → throughput chart (the unique "only-on-AMD" angle).
- Fill placeholders in `blog/yolov8-rocm-mi300x/README.md` (numbers + `images/`), then PR to `github.com/ROCm/rocm-blogs`.

## Guiding angle (don't lose this)
- **Verification-first**: golden reference everywhere, and **re-validate after quantization** (confirm mAP holds).
- **Unique niche**: CV + verification + efficient inference (ROCm/edge) + agentic/MCP.
- **Differentiator vs Hipbridge**: CV kernels — their coverage is transformer row-wise only.

## Parked ideas
- **CV-model-as-an-MCP-tool** demo (agent calls the YOLOv8 detector via MCP).
- `tspeterkim/flash-attention-minimal` kernel port (stretch → second blog).
- 5-stage architecture diagram: train (ROCm) → validate → quantize → serve → MCP.

## Other live threads (not ROCm)
- **Meta** (Build/DevOps, "testing with AI"): interview went well, feedback expected ~Mon. Q&A/STAR prep bank ready to build on request.
- **OLIX** (Staff Software DevOps): deck at `~/Downloads/OLIX_Interview_Deck.{html,pdf}`.
