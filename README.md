# Python3 Full-Stack Machine Learning Workspace for NVIDIA RTX 5000 series GPUs

This repository contains a production-ready Docker environment for **end-to-end machine learning development** on **NVIDIA RTX 5000 series GPUs** in Windows 11. Build complete ML applications from training to deployment with GPU acceleration.

**Python version:** 3.12  
**Base image:** `nvcr.io/nvidia/pytorch:26.03-py3`  
**CUDA version:** 13.2  
**PyTorch version:** 2.11.0a0 (NVIDIA-optimized NGC build)

---

## Development System Specs

This workspace was developed and tested on:

- **OS:** Microsoft Windows 11 Home
- **CPU:** Intel® Core™ i5-13450HX
- **GPU:** NVIDIA® GeForce RTX 5060
- **Storage:** 512 GB SSD
- **Memory:** 16 GB RAM
- **WSL:** Ubuntu 24.04

---

## What's Included

This containerized workspace includes everything you need for full-stack ML development:

- **PyTorch 2.11** with CUDA 13.2 GPU acceleration (NVIDIA-optimized NGC build)
- **torchaudio** compiled from source for Blackwell (sm_120)
- **llama-cpp-python** compiled from source for Blackwell (sm_120)
- **Hugging Face stack** — `transformers`, `accelerate`, `sentence-transformers`
- **Jupyter Lab 4.3.6** for interactive ML development
- **Node.js LTS + npm** for modern frontend development
- **FastAPI + Uvicorn** for ML model serving
- **Vite** for rapid frontend prototyping
- Optimized VRAM management for 8 GB GPUs

---

## Why This Workspace?

### Full-Stack ML Development
Build complete ML applications in one environment:
- **Train** models in Jupyter Lab
- **Deploy** APIs with FastAPI
- **Build** frontends with React/Vue/Svelte + Vite
- **Integrate** everything with GPU acceleration

### Blackwell Architecture Optimization
- NVIDIA-optimized PyTorch and torchvision pre-installed from the NGC base image
- `torchaudio` compiled from source targeting sm_120 (Blackwell)
- Custom-compiled `llama-cpp-python` targeting sm_120
- CUBLAS support for early Blackwell compatibility
- Optimized CUDA kernel loading

### Why PyTorch from NGC and Not PyPI?

`torch` and `torchvision` are intentionally **not** installed via `requirements.txt`. The NGC PyTorch base image ships NVIDIA-optimized builds of both packages. Installing them from PyPI would silently overwrite those builds with generic community wheels and lose Blackwell-specific performance improvements.

Similarly, `torchaudio` is **not** available in any NGC container and is compiled from source in the Dockerfile against the NGC PyTorch build with `TORCH_CUDA_ARCH_LIST=12.0`. A PyPI wheel would try to install its own pinned version of `torch`, replacing the NGC build.

### Why No TensorFlow?

NVIDIA announced that the NGC TensorFlow container line reached end of life at the 25.02 release — no further TF NGC containers will be published. Continuing to build on that base would have frozen the CUDA version at 12.8 with no upgrade path. This workspace migrated to the NGC PyTorch base as the actively maintained, Blackwell-optimized successor.

### Professional Python ML Stack
- PyTorch 2.11, torchvision, torchaudio (Blackwell-optimized)
- Transformers, accelerate, sentence-transformers
- Pandas, scikit-learn, matplotlib, seaborn
- Computer vision: OpenCV
- Document processing: PDF, DOCX, Excel
- Vector search: FAISS

### Modern Frontend Tooling
- Node.js LTS with latest npm
- Vite for blazing-fast HMR
- Full build toolchain for native modules

### 8 GB VRAM Optimization
- Memory fragmentation reduction
- Lazy CUDA module loading
- Controlled batch sizes for llama.cpp

---

## Prerequisites

### Required Software

1. **NVIDIA Driver** (version 595+ for CUDA 13.2 / Blackwell)
2. **WSL2** with **Ubuntu 24.04**
3. **Docker Desktop** (latest, with WSL2 backend)
4. **CUDA Toolkit 13.x** in WSL2

> **Note:** Starting with CUDA 13.1, the Windows display driver is no longer bundled with the CUDA Toolkit installer. You must download and install the driver separately from the [NVIDIA driver download page](https://www.nvidia.com/en-us/drivers/).

### Verification Steps

**Check GPU in WSL2:**
```bash
nvidia-smi
```
Should show your RTX GPU with driver 595+ and CUDA 13.x.

**Verify CUDA Toolkit:**
```bash
nvcc --version
```
Should display CUDA 13.x.

**Test Docker GPU Access:**
```bash
docker run --rm --gpus all nvcr.io/nvidia/cuda:13.2.0-base-ubuntu24.04 nvidia-smi
```

### Need Help Setting Up?

Watch this comprehensive tutorial: **[WSL2 + NVIDIA Driver + CUDA Setup](https://youtu.be/PevhQHn-6R8?si=50_s43qlGhjAzCqC)**

---

## Build and Run

### 1. Build the container (run from the `dockerfiles/` subdirectory)
```bash
docker compose build
```
**Build time:** 30 minutes - 1 hour (torchaudio and llama-cpp-python both compile from source)

### 2. Start the service
```bash
docker compose up -d
```

### 3. Access Jupyter Lab
Open your browser at:
```
http://127.0.0.1:8888/lab?token=rtx-5060_dev
```
**Token:** `rtx-5060_dev`

### 4. Verify GPU Support

Create a new notebook and run:
```python
import torch
import torchaudio
from llama_cpp import llama_cpp

# PyTorch
print(f"PyTorch CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Device: {torch.cuda.get_device_name(0)}")
print(f"CUDA Version: {torch.version.cuda}")

# torchaudio
print(f"torchaudio version: {torchaudio.__version__}")

# llama-cpp-python
print(f"llama-cpp GPU Support: {llama_cpp.llama_supports_gpu_offload()}")
```

Expected output:
```
PyTorch CUDA Available: True
GPU Device: NVIDIA GeForce RTX 5060
CUDA Version: 13.2
torchaudio version: 2.x.x+cu132
llama-cpp GPU Support: True
```

### 5. Shutdown the container

To stop and remove the container:
```bash
docker compose down
```

---

## Project Structure
```
python3-workspace/
├── dockerfiles/
│   ├── Dockerfile
│   ├── rtx-5060_dev-requirements.txt
│   └── docker-compose.yml
└── README.md
```

---

## Configuration Details

### Dockerfile Overview

| Section | Description |
|---------|-------------|
| **Base Image** | NVIDIA PyTorch 26.03 with CUDA 13.2 |
| **System Deps** | Build tools, OpenGL, graphviz, Node.js |
| **Python Link** | Ensures `python` command works |
| **Node.js** | LTS via NodeSource with latest npm |
| **CUDA Env** | CUDA_HOME, PATH, LD_LIBRARY_PATH |
| **Build Args** | `FORCE_CUBLAS=on` for Blackwell |
| **Dependencies** | Python packages (torch/torchvision excluded — NGC build) |
| **torchaudio** | Source build for sm_120 with NGC torch |
| **llama-cpp** | Source build for sm_120 architecture |
| **Runtime Opts** | Lazy CUDA module loading |
| **Verification** | C-level CUDA + llama-cpp GPU support check |
| **Ports** | 8888 (Jupyter), 5173 (Vite), 8000 (API) |

### Pip Constraints File

The NGC PyTorch 26.03 base image ships a pip constraints file at `/etc/pip/constraint.txt` that locks all Python packages used during image creation. This prevents unintentional overwrites of NVIDIA-optimized builds when you install additional packages.

If you need a different version of a constrained package, edit `/etc/pip/constraint.txt` inside the container to remove the relevant pin before installing.

### docker-compose.yml Features

- **GPU Access:** All GPUs with compute, utility, video capabilities
- **VRAM Optimization:** Expandable segments (`PYTORCH_ALLOC_CONF`), controlled llama.cpp batching (`LLAMA_ARG_BATCH`)
- **Port Mapping:** Jupyter (8888), Vite (5173), FastAPI (8000)
- **Volumes:** Workspace mount + persistent Jupyter settings
- **Resource Limits:** IPC host, unlimited memlock, 64 MB stack
- **Auto-verification:** GPU check before Jupyter launch

### Python Dependencies

**Core ML/Data Science:**
- pandas, scikit-learn, matplotlib, seaborn
- faiss-cpu (vector search)

**Deep Learning:**
- torch, torchvision — pre-installed in NGC base image (not in requirements.txt)
- torchaudio — compiled from source in Dockerfile (not in requirements.txt)
- transformers, accelerate, sentence-transformers

**Computer Vision/Documents:**
- opencv-python, pdf2image, pypdf
- pytesseract, python-docx, openpyxl
- beautifulsoup4

**Backend/API:**
- fastapi, uvicorn[standard], pydantic-settings
- python-multipart, websockets, watchfiles
- python-dotenv

**Utilities:**
- cmake, pydot, markdown
- hf-xet, ipywidgets

> **Note:** `llama-cpp-python` is compiled separately in the Dockerfile with custom CMAKE flags (`GGML_CUDA=on`, `CMAKE_CUDA_ARCHITECTURES=120`). It is not listed in `requirements.txt`.

---

## Exposed Ports

| Port | Service | Access |
|------|---------|--------|
| **8888** | Jupyter Lab | `http://localhost:8888/lab?token=rtx-5060_dev` |
| **5173** | Vite Dev Server | `http://localhost:5173` |
| **8000** | FastAPI/Uvicorn | `http://localhost:8000/docs` |

### Using the Ports

**Jupyter Lab (Default):**
Already running when container starts.

**Vite Frontend:**
```bash
docker exec -it rtx-5060_dev bash
cd /workspace/my-frontend
npm run dev -- --host 0.0.0.0
```
Access at `http://localhost:5173`

**FastAPI Server:**
```bash
docker exec -it rtx-5060_dev bash
cd /workspace/api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Access at `http://localhost:8000/docs`

---

## Full-Stack ML Workflow

### Complete Application Example

**Step 1: Train Model (Jupyter Lab - Port 8888)**
```python
import torch
import torch.nn as nn

# Define and train your model
model = YourModel()
# ... training code ...

# Save model
torch.save(model.state_dict(), '/workspace/models/model.pth')
print("Model saved!")
```

**Step 2: Serve Model (FastAPI - Port 8000)**

Create `/workspace/backend/main.py`:
```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI()

# Load model
model = YourModel()
model.load_state_dict(torch.load('/workspace/models/model.pth'))
model.eval()

class PredictionRequest(BaseModel):
    data: list

@app.post("/predict")
async def predict(request: PredictionRequest):
    with torch.no_grad():
        input_tensor = torch.tensor(request.data)
        result = model(input_tensor)
    return {"prediction": result.tolist()}

@app.get("/health")
async def health():
    return {"status": "healthy", "gpu": torch.cuda.is_available()}
```

Run server:
```bash
docker exec -it rtx-5060_dev bash
cd /workspace/backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Step 3: Build Frontend (Vite - Port 5173)**

Create frontend:
```bash
docker exec -it rtx-5060_dev bash
cd /workspace
npm create vite@latest frontend -- --template react
cd frontend
npm install
npm run dev -- --host 0.0.0.0
```

Fetch predictions in your React app:
```javascript
// src/App.jsx
import { useState } from 'react'

function App() {
  const [prediction, setPrediction] = useState(null)

  const getPrediction = async () => {
    const response = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data: [1, 2, 3, 4] })
    })
    const result = await response.json()
    setPrediction(result.prediction)
  }

  return (
    <div>
      <button onClick={getPrediction}>Get Prediction</button>
      {prediction && <p>Result: {prediction}</p>}
    </div>
  )
}

export default App
```

**Access Everything:**
- Training: `http://localhost:8888`
- API: `http://localhost:8000/docs`
- Frontend: `http://localhost:5173`

---

## Advanced Configuration

### Rebuild with Different CUBLAS Setting

Test native Blackwell kernels (may not work on all driver versions):
```bash
docker compose build --build-arg FORCE_CUBLAS=off --no-cache
```

Revert to CUBLAS (default, recommended):
```bash
docker compose build --build-arg FORCE_CUBLAS=on --no-cache
```

### Custom Jupyter Configuration

Edit `docker-compose.yml`:
```yaml
command: >
  bash -c "python3 /usr/local/bin/verify_gpu.py &&
  echo '' &&
  jupyter lab
  --ip 0.0.0.0
  --allow-root
  --no-browser
  --log-level=INFO
  --ServerApp.token='rtx-5060_dev'
  --ResourceUseDisplay.track_cpu_percent=True"
```

### Install Additional Python Packages

**Temporary:**
```bash
docker exec -it rtx-5060_dev pip install package-name
```

**Permanent:**
1. Add to `rtx-5060_dev-requirements.txt`
2. Rebuild: `docker compose build`

> **Heads-up:** The NGC base image includes a pip constraints file at `/etc/pip/constraint.txt`. If a package install fails due to a version conflict, check whether the package is pinned there and remove the relevant line before retrying.

### Install Node.js Packages Globally
```bash
docker exec -it rtx-5060_dev npm install -g package-name
```

### Monitor GPU Usage
```bash
docker exec -it rtx-5060_dev watch -n 1 nvidia-smi
```

---

## Troubleshooting

### "CUDA out of memory" Errors

**Solutions:**
1. Reduce batch size
2. Enable gradient checkpointing:
```python
model.gradient_checkpointing_enable()
```
3. Use mixed precision (BF16 — preferred on Blackwell):
```python
from torch.amp import autocast, GradScaler
scaler = GradScaler()
with autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(input)
```
4. Clear cache:
```python
torch.cuda.empty_cache()
```
5. Monitor usage: `nvidia-smi`

### llama-cpp-python Not Using GPU

Run verification:
```bash
docker exec -it rtx-5060_dev python3 /usr/local/bin/verify_gpu.py
```

Expected output:
```
Detected: NVIDIA GeForce RTX 5060 | sm_120
Testing llama-cpp GPU backend...
llama-cpp GPU Support: True
Hardware verification passed!
```

If failed, rebuild:
```bash
docker compose build --no-cache
```

### Container Won't Start

**Checklist:**
1. Docker Desktop WSL2 integration enabled
2. `nvidia-smi` works in WSL2
3. Driver version is 595 or newer
4. Ports 8888, 5173, 8000 are available
5. Check logs: `docker compose logs -f`

### Vite Not Accessible

Ensure server binds to `0.0.0.0`:

**Command line:**
```bash
npm run dev -- --host 0.0.0.0
```

**vite.config.js:**
```javascript
export default {
  server: {
    host: '0.0.0.0',
    port: 5173
  }
}
```

### Package Version Conflicts After `pip install`

The NGC 26.03 base image ships a pip constraints file at `/etc/pip/constraint.txt`. If you hit a version conflict, inspect the file and remove the pin for the affected package:

```bash
# Inside the container
cat /etc/pip/constraint.txt
```

Then install your package without the conflicting constraint. To make the change permanent, add a step to the Dockerfile that removes the relevant line before the pip install step.

---

## Performance Optimization

### For 8 GB VRAM Cards

**1. Gradient Accumulation**
```python
# Instead of batch_size=32
# Use batch_size=8 with 4 accumulation steps
accumulation_steps = 4
for i, (inputs, labels) in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, labels) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**2. Gradient Checkpointing**
```python
from torch.utils.checkpoint import checkpoint

class MyModel(nn.Module):
    def forward(self, x):
        x = checkpoint(self.expensive_layer, x)
        return x
```

**3. Mixed Precision Training (BF16 — recommended for Blackwell)**
```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()

    with autocast(device_type='cuda', dtype=torch.bfloat16):
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### PyTorch Settings
```python
# Enable TF32 for Blackwell (faster matrix math with minimal precision cost)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Enable cudNN autotuner (best for fixed input sizes)
torch.backends.cudnn.benchmark = True
```

### Environment Variables

Already configured in `docker-compose.yml`:
```yaml
CUDA_MODULE_LOADING=LAZY                         # Defers kernel loading; reduces cold-start latency on Blackwell
NVIDIA_DISABLE_REQUIRE=1                         # Bypasses container-runtime driver version gate
PYTORCH_ALLOC_CONF=expandable_segments:True      # Reduces VRAM fragmentation on 8 GB cards
LLAMA_ARG_BATCH=512                              # Caps llama.cpp batch allocation; prevents OOM on 8 GB VRAM
```

---

## Security Notes

### Jupyter Lab Token

Default token: `rtx-5060_dev`

**Change for production:**
```yaml
# docker-compose.yml
--ServerApp.token='your-secure-token'
```

**Or use password:**
```bash
docker exec -it rtx-5060_dev jupyter lab password
```

### Port Binding

Restrict to localhost only:
```yaml
ports:
  - "127.0.0.1:8888:8888"
  - "127.0.0.1:5173:5173"
  - "127.0.0.1:8000:8000"
```

---

## Additional Resources

### Documentation
- [NVIDIA PyTorch Container (NGC)](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
- [PyTorch 26.03 Release Notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-03.html)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Vite](https://vitejs.dev/)

### Related Projects
- **Simple ML Workspace:** [python3-workspace](https://github.com/kennethtomaniog2035/rtx5060-wsl2-workspace) — lightweight version without frontend tools and language model dependencies

---

## Support

Having issues?

1. Review logs: `docker compose logs -f`
2. Verify prerequisites: `nvidia-smi`, `nvcc --version`
3. Open GitHub issue with error logs and system info

---

## System Requirements Summary

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | RTX 5060 (8 GB) | RTX 5070+ (12 GB+) |
| **RAM** | 16 GB | 32 GB |
| **Storage** | 50 GB free | 100 GB+ SSD |
| **Driver** | 595+ | Latest |
| **CUDA** | 13.2 | 13.x |
| **OS** | WSL2 Ubuntu 24.04 | Same |
