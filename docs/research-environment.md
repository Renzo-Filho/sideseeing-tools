# Research environment on `utk-desktop-01`

This is a user-owned Conda environment. It requires no sudo and uses the RTX 3070 directly on the host. Run these commands **on desktop-01**, from a checkout of this repository. The Python 3.12 environment, CUDA PyTorch GPU detection, Whisper and library imports, SAM3 image class imports, and `pip check` have been tested on desktop-01. SAM3 model inference has not yet been tested there.

## Install Conda if needed

If `conda --version` works, initialize it for the current Bash session with `eval "$(conda shell.bash hook)"` and skip to the next section. Otherwise, on the current Ubuntu x86-64 machine:

```bash
curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o "$HOME/Miniforge3-Linux-x86_64.sh"
bash "$HOME/Miniforge3-Linux-x86_64.sh" -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda --version
conda init bash
conda config --set auto_activate_base false
```

Reconnect after `conda init bash`. In later SSH sessions, `conda activate sideseeing` will work directly. If Conda was installed elsewhere, initialize that installation instead.

## Create the environment

```bash
cd "$HOME/GitHub/sideseeing-tools"  # adjust to the actual clone location
conda env create -f environment.yml
conda activate sideseeing
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[vision,dev]'
python -m pip install openai-whisper
python -m pip check
```

The Conda packages cover FFmpeg and native geospatial libraries used by the current package; the subsequent pip steps install CUDA PyTorch, this checkout, SAM3's current Hugging Face image backend, and Whisper. `environment.yml` does not include the pip steps so the CUDA wheel index and install order are explicit. This file is a bootstrap specification, not yet a tested lockfile.

The current `Segmenter` downloads the gated `facebook/sam3` checkpoint through Hugging Face. Obtain access at <https://huggingface.co/facebook/sam3> and authenticate on desktop-01 with `hf auth login` before the first model run. No token should be committed to Git. The existing library uses Transformers for SAM3 images. Transformers also has a SAM3 video API that we can evaluate in this same environment. The official `facebookresearch/sam3` package is an alternative video backend; it is not installed here.

## Check the setup

```bash
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")'
python -c 'import sideseeing_tools, whisper; from transformers import Sam3Model, Sam3Processor; print("imports OK")'
ffmpeg -version | head -n 1
python -m pip check
```

On desktop-01, `torch.cuda.is_available()` should be `True` and the GPU should be the RTX 3070. If CUDA is unavailable, capture these four outputs together with `nvidia-smi` before changing package versions.

For a first Whisper run on a short video, use a model that fits comfortably in 8 GB and keep inference jobs sequential:

```bash
whisper /path/to/short-video.mp4 --model small --device cuda --output_dir /path/to/output
```

The first run downloads model weights. Start SAM3 image inference with `batch_size=1` on a few frames and record peak GPU memory before increasing batch size. A SAM3 video tracking backend should be evaluated separately because continuous tracking has different memory and output behavior from the library's current independent-frame segmentation.

## Source documentation

- [Miniforge installation](https://github.com/conda-forge/miniforge)
- [SAM3 prerequisites and video predictor](https://github.com/facebookresearch/sam3/blob/main/README.md)
- [Hugging Face SAM3 image API](https://huggingface.co/docs/transformers/model_doc/sam3)
- [Whisper install, FFmpeg, and model sizes](https://github.com/openai/whisper/blob/main/README.md)
- [Whisper Python 3.12 package metadata](https://github.com/openai/whisper/blob/main/pyproject.toml)
