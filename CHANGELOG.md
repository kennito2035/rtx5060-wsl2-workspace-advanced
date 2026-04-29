# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] — April 2026

### Breaking Changes

- **Base image replaced:** Switched from `nvcr.io/nvidia/tensorflow:25.02-tf2-py3` to `nvcr.io/nvidia/pytorch:26.03-py3`. The TensorFlow NGC container was found to have unstable and incomplete RTX 50-series (sm_120) support. The NGC PyTorch 26.03 image ships a significantly more mature Blackwell GPU stack and is now the foundation of this workspace.
- **CUDA upgraded from 12.8 → 13.2.** Packages and constraints have been updated accordingly. Some third-party libraries may still exhibit compatibility issues as CUDA 13.2 / Blackwell support matures across the ecosystem.
- **PyTorch is now the primary pre-installed framework.** `torch`, `torchvision`, and `torchaudio` are supplied as NVIDIA-optimized builds by the base image and must not be reinstalled from PyPI. TensorFlow is no longer included by default.

### Added

- `NVIDIA_DISABLE_REQUIRE=1` environment variable in `docker-compose.yml` to skip minor-version compatibility checks between pip packages and the CUDA 13.2 system libraries.
- `hf-xet` added to requirements for fast large-file transfers from Hugging Face Hub.
- `scikit-build-core` pre-install step before the `llama-cpp-python` source build. This is required when using `--no-build-isolation`; without it pip cannot import the build backend declared in `llama-cpp-python`'s `pyproject.toml` and aborts with `BackendUnavailable`.
- Manipulation of the NGC global pip constraint file (`/etc/pip/constraint.txt`) to strip `numpy`, `ml-dtypes`, and `packaging` entries before applying project-local pins, preventing silent overrides during dependency resolution.
- `packaging<=25.0` pin to satisfy `nvidia-dali-cuda130 2.0.0`, which declares incompatibility with `packaging 26+`.

### Changed

- **`llama-cpp-python` build now uses `--no-build-isolation`** so that the already-installed NumPy is reused rather than downloaded into an isolated build environment, eliminating the repeated uninstall/reinstall churn seen in prior builds.
- **`ml-dtypes` constraint relaxed** from `==0.4.1` to `>=0.5.0,<0.6.0`. This satisfies `onnx-ir 0.2.0` (which requires `ml_dtypes>=0.5.0`) while still blocking `0.6+`, which begins requiring NumPy 2.x.
- **`pip`, `setuptools`, and `wheel` upgrade order changed.** `pip` is now upgraded first in its own step. `setuptools` and `wheel` use `--ignore-installed` because the NGC base installs them via Debian (no `RECORD` file), which caused pip's uninstall step to abort the entire `RUN` layer in v1.
- `verify_gpu.py` smoke test updated to validate PyTorch GPU access and `llama-cpp` CUDA support. The previous TensorFlow-based check has been removed.
- `NumPy 1.26.4` is now force-pinned at the end of the dependency install as a final step to prevent any transitive dependency from silently upgrading it to 2.x.

### Removed

- `tomli` removed from requirements — `tomllib` has been available in the Python 3.12 standard library since its release and a separate package is no longer needed.
- `httptools` removed as a standalone entry — it is already pulled in transitively by `uvicorn[standard]`.

---

## [1.0.0] — December 2025

- Initial release.
- Base image: `nvcr.io/nvidia/tensorflow:25.02-tf2-py3` (CUDA 12.8, Python 3.12.3).
- TensorFlow 2.x + PyTorch with CUDA 12.8 GPU acceleration.
- `llama-cpp-python` compiled from source targeting Blackwell sm_120 with `FORCE_CUBLAS=on`.
- Jupyter Lab, FastAPI/Uvicorn, Node.js LTS, and Vite dev server included.
- Ports exposed: 8888 (Jupyter), 5173 (Vite), 8000 (FastAPI).
- VRAM optimizations for 8 GB cards: expandable segments, lazy CUDA module loading, controlled llama.cpp batch size.
