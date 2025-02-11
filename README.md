## MARMamba

This is the source code for MARMamba (MARMamba: A Single-Domain CT Metal Artifact Reduction Model Based on Multi-Scale Mamba).

### Architecture

![Model Architecture](./Figures/Backbone.png)

### Requirement

The project is built with Python 3.12.3, CUDA 12.1 and Mamba 2.2.4. Using the following command to install dependency packages:

```
pip install -r requirements.txt
```

Specifically, our project requires the following libraries:

```
einops==0.8.1
h5py==3.12.1
lpips==0.1.4
mamba_ssm==2.2.4
matplotlib==3.10.0
nibabel==5.3.2
numpy==2.2.2
opencv_python==4.11.0.86
Pillow==11.1.0
scipy==1.15.1
thop==0.1.1.post2209072238
timm==1.0.14
torch==2.3.0+cu121
torchvision==0.18.0+cu121
```

### Checkpoint

MARMamba checkpoint is available at path `checkpoint\MARMamba_ckpt`

