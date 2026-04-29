###############################################################################
# 1.) Base: NVIDIA PyTorch 26.03 (CUDA 13.2, sm_120 native)
#
# Switched from nvcr.io/nvidia/tensorflow:25.02-tf2-py3.
# Reason: Unstable / incomplete RTX 50-series (sm_120)
# support observed in current TensorFlow releases.
#
# NVIDIA PyTorch 26.03 includes newer CUDA and
# PyTorch builds with significantly better support
# for RTX 5060 / Blackwell GPUs compared to the
# current TensorFlow container ecosystem.
#
# NOTE:
# CUDA 13.2 and Blackwell support are still relatively
# new across the ML ecosystem, so some third-party
# libraries may still exhibit compatibility issues.
###############################################################################
FROM nvcr.io/nvidia/pytorch:26.03-py3

###############################################################################
# 2.) Working directory
###############################################################################
WORKDIR /workspace

###############################################################################
# 3.) System-level dependencies
###############################################################################
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential cmake git curl python3-dev python3-pip \
        graphviz libgl1 libglib2.0-0 \
        libgomp1 \
        ffmpeg libsndfile1-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

###############################################################################
# 4.) Ensure python symlink exists (forced)
###############################################################################
RUN ln -sf /usr/bin/python3 /usr/bin/python

###############################################################################
# 5.) Install Node.js
###############################################################################
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g npm@latest

###############################################################################
# 6.) Build-time environment
###############################################################################
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}

###############################################################################
# 7.) CUDA build policy toggle
# Default ON for early Blackwell (sm_120).
# Set to "off" later to test llama.cpp native kernels.
###############################################################################
ARG FORCE_CUBLAS=on

###############################################################################
# 8.) Copy requirements for caching
###############################################################################
COPY dockerfiles/rtx-5060_dev-requirements.txt .

###############################################################################
# 9.) Build python dependencies for sm_120
#
# NGC PyTorch 25.03+ ships /etc/pip/constraint.txt which is applied
# globally by pip. numpy, ml-dtypes, and packaging entries are stripped
# from it first so our explicit pins below are not silently overridden.
#
# Exclusion list: llama-cpp-python (built from source below),
# numpy and ml-dtypes (force-pinned at the end to prevent 2.x
# from sneaking in via transitive deps).
#
# packaging is pinned to <=25.0 to satisfy nvidia-dali-cuda130 2.0.0,
# which declares an incompatibility with packaging 26+.
#
# ml-dtypes is relaxed from ==0.4.1 to >=0.5.0,<0.6.0 so that
# onnx-ir 0.2.0 (which requires ml_dtypes>=0.5.0) is satisfied while
# still blocking 0.6+ which begins requiring numpy 2.x.
#
# llama-cpp-python is built with --no-build-isolation so it reuses
# the already-installed numpy instead of downloading it a second time,
# preventing the repeated uninstall/reinstall churn seen in prior builds.
#
# pip is upgraded first alone. setuptools and wheel use --ignore-installed
# because the NGC base installs them via Debian (no RECORD file), which
# causes pip's uninstall step to fail and abort the entire RUN layer.
###############################################################################
RUN python3 -m pip install --upgrade pip && \
    pip install --upgrade setuptools wheel --ignore-installed && \
    sed -i '/^numpy/d; /^ml-dtypes/d; /^packaging/d' /etc/pip/constraint.txt && \
    echo "numpy==1.26.4"            > /tmp/constraints.txt && \
    echo "ml-dtypes>=0.5.0,<0.6.0" >> /tmp/constraints.txt && \
    echo "packaging<=25.0"          >> /tmp/constraints.txt && \
    \
    # I. Install main requirements (excluding source-built packages)
    grep -vE "llama-cpp-python|numpy|ml-dtypes" rtx-5060_dev-requirements.txt > /tmp/reqs.txt || true && \
    python3 -m pip install --no-cache-dir -c /tmp/constraints.txt -r /tmp/reqs.txt && \
    \
    # II. Build llama-cpp-python from source for sm_120.
    #     --no-build-isolation reuses the already-installed numpy rather than
    #     downloading a fresh copy into an isolated build env, preventing churn.
    #     scikit-build-core must be pre-installed before using --no-build-isolation;
    #     otherwise pip cannot import the build backend declared in
    #     llama-cpp-python's pyproject.toml and aborts with BackendUnavailable.
    python3 -m pip install --no-cache-dir scikit-build-core && \
    export CMAKE_ARGS="-DGGML_CUDA=on \
                       -DGGML_CUDA_FORCE_CUBLAS=${FORCE_CUBLAS} \
                       -DCMAKE_CUDA_ARCHITECTURES=120 \
                       -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc" && \
    export FORCE_CMAKE=1 && \
    python3 -m pip install llama-cpp-python --no-cache-dir -c /tmp/constraints.txt \
        --force-reinstall --no-build-isolation && \
    \
    # III. Force-pin final versions so no transitive dep overwrites them
    python3 -m pip install --no-cache-dir \
        "numpy==1.26.4" \
        "ml-dtypes>=0.5.0,<0.6.0" \
        "packaging<=25.0" \
        --force-reinstall

###############################################################################
# 10.) Build torchaudio from source for sm_120
#
# torchaudio is not shipped in any NGC PyTorch container.
# A plain `pip install torchaudio` pulls a community wheel
# with a hard torch== pin that would silently replace the
# NVIDIA-optimized torch build — the same risk the requirements
# file warns about for torch/torchvision.
#
# We clone main (no stable tag exists for torch 2.11.0a0),
# compile with CUDA targeting Blackwell (sm_120), and install
# with --no-deps so the NGC torch is never touched.
#
# --no-build-isolation reuses the already-installed torch
# headers rather than pulling a generic torch into an isolated
# build env, which is essential for ABI compatibility.
###############################################################################
RUN git clone --depth 1 https://github.com/pytorch/audio.git /tmp/audio && \
    cd /tmp/audio && \
    USE_CUDA=1 \
    USE_CUDNN=1 \
    TORCH_CUDA_ARCH_LIST="12.0" \
    python3 -m pip install \
        --no-build-isolation \
        --no-deps \
        --no-cache-dir \
        . && \
    rm -rf /tmp/audio

###############################################################################
# 11.) Runtime optimization
#
# LD_LIBRARY_PATH is intentionally not re-declared here.
# Step 6 already sets it to include both the CUDA lib64 path
# and the stubs path (needed at build time for source compilations
# in Steps 9 and 10). Re-expanding ${LD_LIBRARY_PATH} here would
# only duplicate the prefix and leave stubs in place anyway, so
# Step 6's value is the correct final state for this dev image.
###############################################################################
ENV CUDA_MODULE_LOADING=LAZY

###############################################################################
# 12.) Verification (smoke test) — checks PyTorch GPU + llama-cpp GPU + torchaudio
###############################################################################
RUN echo "import torch; \
import torchaudio; \
import llama_cpp.llama_cpp as llm; \
\
cap = torch.cuda.get_device_capability(0); \
print(f'Detected: {torch.cuda.get_device_name(0)} | sm_{cap[0]}{cap[1]}'); \
assert cap[0] >= 12, 'Blackwell GPU not found'; \
\
backends = getattr(torchaudio, 'list_audio_backends', lambda: ['N/A (removed in torchaudio>=2.9)'])(); \
print(f'torchaudio {torchaudio.__version__} | backends: {backends}'); \
\
print('Testing llama-cpp GPU backend...'); \
has_gpu = llm.llama_supports_gpu_offload(); \
print(f'llama-cpp GPU Support: {has_gpu}'); \
assert has_gpu, 'llama-cpp was built without CUDA support!'; \
\
print('Hardware verification passed!')" > /usr/local/bin/verify_gpu.py

###############################################################################
# 13.) Ports exposed to the host
###############################################################################
EXPOSE 8888 5173 8000

###############################################################################
# 14.) Default entry
###############################################################################
CMD ["bash"]