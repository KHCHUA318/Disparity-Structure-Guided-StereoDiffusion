# DSG-SD: Disparity-Structure Guided StereoDiffusion

DSG-SD is a training-free method for single-image stereo right-view synthesis. Given one left-view image, the pipeline generates a stereo pair using a frozen Stable Diffusion model, a frozen DPT depth model, DDIM inversion and inference-time latent operations.

The implementation keeps all pretrained model weights unchanged. The proposed components are applied only during inference.

## Main Components

DSG-SD contains three active components:

1. **Structure-guided disparity refinement (`D'`)**  
   Refines the initial disparity cue before Stereo Pixel Shift. It strengthens weak but structure-connected regions while keeping the global disparity layout controlled.

2. **Depth-boundary-aware SPSMD masking**  
   Suppresses SPSMD copying near strong depth boundaries so that latent refreshing is kept in safer regions and reduced around foreground-background transitions.

3. **Structure-aware latent alignment**  
   Slightly strengthens disparity-induced latent movement in safe image-structure regions while suppressing movement near depth boundaries.

The original StereoDiffusion attention-sharing mechanism is retained.

## Package Structure

```text
DSG-SD/
├── img2stereo.py              # main inference script
├── stereoutils.py             # core stereo utilities and DSG-SD components
├── ptp_utils.py               # diffusion inversion and sampling utilities
├── seq_aligner.py             # token alignment helper
├── inspect_stereodiffusion.py # optional inspection helper
├── sitecustomize.py           # runtime compatibility helper
├── requirements.txt           # Python package list
├── run_demo.sh                # Linux/macOS demo command
├── run_demo.bat               # Windows demo command
├── DPT/                       # place DPT source code folder contents here
├── stablediffusion/           # place Stable Diffusion LDM source code folder contents here
├── midas_models/              # place DPT checkpoint here
├── data/
│   ├── kitti2015/
│   │   ├── left/              # selected KITTI2015 left images
│   │   └── right/             # corresponding right-view ground truth
│   └── middlebury2014/
│       ├── left/              # selected Middlebury 2014 left images
│       └── right/             # corresponding right-view ground truth
└── outputs/                   # generated stereo outputs are saved here
```

## Pretrained Files to Add

Before running the demo, copy the pretrained support folders into the DSG-SD root directory.

The expected final structure is:

```text
DSG-SD/
├── DPT/
│   └── dpt/
│       └── models.py
├── stablediffusion/
│   └── ldm/
│       └── models/
│           └── diffusion/
│               └── ddim.py
└── midas_models/
    └── dpt_hybrid-midas-501f0c75.pt
```

The demo scripts use this default DPT checkpoint path:

```text
midas_models/dpt_hybrid-midas-501f0c75.pt
```

Stable Diffusion v1-4 is loaded through the Diffusers pipeline in `img2stereo.py`. The runtime should have access to the pretrained Stable Diffusion v1-4 weights through local cache or normal Diffusers loading.

## Environment Setup

Install the required Python packages in a suitable PyTorch environment:

```bash
pip install -r requirements.txt
```

A CUDA GPU is recommended.

## Quick Demo

### Linux / macOS

```bash
bash run_demo.sh
```

Or provide a custom DPT checkpoint path:

```bash
bash run_demo.sh /path/to/dpt_hybrid-midas-501f0c75.pt
```

### Windows Command Prompt

```bat
run_demo.bat
```

Or provide a custom DPT checkpoint path:

```bat
run_demo.bat C:\path\to\dpt_hybrid-midas-501f0c75.pt
```

The demo uses this included KITTI2015 image:

```text
data/kitti2015/left/000187_10.png
```

The generated stereo output is saved in:

```text
outputs/000187_10.png
```

## Manual Inference Command

Run DSG-SD on any included or custom left-view image:

```bash
python img2stereo.py \
  --img_path data/kitti2015/left/000187_10.png \
  --depthmodel_path midas_models/dpt_hybrid-midas-501f0c75.pt \
  --direction uni
```

Example using a Middlebury image:

```bash
python img2stereo.py \
  --img_path data/middlebury2014/left/Backpack.png \
  --depthmodel_path midas_models/dpt_hybrid-midas-501f0c75.pt \
  --direction uni
```

## Ablation Flags

The three DSG-SD components can be switched on or off using environment variables:

```bash
DSG_USE_DPRIME=1      # enable structure-guided disparity refinement
DSG_USE_SPSMDMASK=1   # enable depth-boundary-aware SPSMD masking
DSG_USE_ALIGN=1       # enable structure-aware latent alignment
```

Set a value to `0` to disable that component for ablation.

Example on Linux/macOS:

```bash
DSG_USE_DPRIME=1 DSG_USE_SPSMDMASK=0 DSG_USE_ALIGN=1 python img2stereo.py \
  --img_path data/kitti2015/left/000187_10.png \
  --depthmodel_path midas_models/dpt_hybrid-midas-501f0c75.pt \
  --direction uni
```

Example on Windows Command Prompt:

```bat
set DSG_USE_DPRIME=1
set DSG_USE_SPSMDMASK=0
set DSG_USE_ALIGN=1
python img2stereo.py --img_path data\kitti2015\left\000187_10.png --depthmodel_path midas_models\dpt_hybrid-midas-501f0c75.pt --direction uni
```

## Output and Evaluation

The generated result is saved as a stereo image containing the reconstructed left view and the generated right view side by side. For quantitative evaluation, split the output into left and right halves, then compare only the generated right view with the ground-truth right view.

The included ground-truth right images are provided for checking outputs and reproducing the reported evaluation setting.

## Notes

- DSG-SD is training-free; no Stable Diffusion, VAE, text encoder, U-Net, or DPT weights are fine-tuned.
- The method is designed for rectified stereo-style right-view synthesis where the main viewpoint difference is horizontal displacement.
- Included data are small selected subsets used for demonstration and evaluation.
