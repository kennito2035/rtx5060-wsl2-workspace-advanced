# Changelog

All notable changes to this project will be documented in this file.

---

## [2.0.0] — 2026-04-30

### Breaking Changes

- **Base image replaced.** Migrated from `nvcr.io/nvidia/tensorflow:25.02-tf2-py3` (CUDA 12.8) to
  `nvcr.io/nvidia/pytorch:26.03-py3` (CUDA 13.2). The container is now PyTorch-first; TensorFlow
  is no longer part of the stack. This change was also driven by NVIDIA's end-of-life announcement
  for the NGC TensorFlow container line: 25.02 was the final release, and no further TF NGC
  containers will be published.
- **`torch`, `torchvision`, and `torchaudio` removed from `requirements.txt`.** `torch` and
  `torchvision` now come pre-installed as NVIDIA-optimized builds from the NGC base image.
  Installing them from PyPI would silently replace those builds with generic wheels. `torchaudio`
  is compiled from source in the Dockerfile with `TORCH_CUDA_ARCH_LIST=12.0` (Blackwell / sm_120).
- **`pypdf2` replaced by `pypdf`.** `PyPDF2` is deprecated upstream; `pypdf` is its maintained
  successor.
- **`LLAMA_ARG_N_BATCH` renamed to `LLAMA_ARG_BATCH`** in `docker-compose.yml` to match the
  correct llama-cpp-python environment variable name.

---

### Dockerfile

- **Base image upgraded** from `nvcr.io/nvidia/tensorflow:25.02-tf2-py3` to
  `nvcr.io/nvidia/pytorch:26.03-py3`, bumping the bundled CUDA runtime from 12.8 → 13.2.
- **Step 10 added:** `torchaudio` is now compiled from source against the NGC PyTorch build with
  `TORCH_CUDA_ARCH_LIST=12.0` to preserve Blackwell-specific optimizations.
- **`TF_ENABLE_ONEDNN_OPTS=0` removed** from runtime environment (no longer relevant without
  TensorFlow).
- **Build-step filtering updated:** the `grep` exclusion list in Step 9 no longer needs to exclude
  `tensorflow`, `numpy`, or `ml-dtypes` constraints that were required to keep TF's compatibility
  layer intact.
- **Smoke-test script (`verify_gpu.py`) updated** to reflect the PyTorch-only stack and confirm
  llama-cpp CUDA support via `llama_supports_gpu_offload()`.

---

### `docker-compose.yml`

- **Service renamed** from `tf-lab` to `ml-workspace` to reflect the shift away from TensorFlow.
- **`LLAMA_ARG_N_BATCH` corrected to `LLAMA_ARG_BATCH`** — the previous key was silently ignored
  by llama-cpp-python.
- Expanded inline comments throughout for clarity on Blackwell ISA, VRAM constraints, and
  `NVIDIA_DISABLE_REQUIRE`.

---

### `requirements.txt`

#### Added
- `transformers` — explicit pin added to ensure a consistent Hugging Face ecosystem version
  alongside `accelerate` and `sentence-transformers`.
- `ipywidgets` — interactive Jupyter widget support.
- `uvicorn[standard]` — replaces bare `uvicorn`; the `[standard]` extra pulls in `uvloop` and
  `httptools`, making the separate `httptools` entry redundant.

#### Removed
- `torch`, `torchvision` — pre-installed in the NGC PyTorch base image; installing from PyPI
  would overwrite NVIDIA-optimized builds.
- `torchaudio` — compiled from source in the Dockerfile; a PyPI wheel would pin a hard
  `torch==` version and replace the NGC build.
- `pypdf2` — deprecated; superseded by `pypdf` (see Breaking Changes).
- `httptools` — now pulled in transitively by `uvicorn[standard]`.
- `tomli` — removed; Python 3.12 ships `tomllib` in the standard library, making this package
  unnecessary.

---

## [1.0.0] — 2025-12-01

Initial release.

- Base image: `nvcr.io/nvidia/tensorflow:25.02-tf2-py3` (CUDA 12.8, Python 3.12).
- PyTorch stack (`torch`, `torchvision`, `torchaudio`) installed from PyPI alongside TensorFlow.
- llama-cpp-python compiled from source with `CMAKE_CUDA_ARCHITECTURES=120` and optional
  `GGML_CUDA_FORCE_CUBLAS` build arg for early Blackwell compatibility.
- Jupyter Lab exposed on port 8888 with GPU smoke test on container start.
- FastAPI / Uvicorn on port 8000, Vite dev server on port 5173.
