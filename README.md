# M3D-NCA: Robust 3D Segmentation with Built-in Quality Contro
### John Kalkhof, Anirban Mukhopadhyay
__https://arxiv.org/pdf/2309.02954.pdf__

> __Abstract:__ Medical image segmentation relies heavily on large-scale deep learning models, such as UNet-based architectures. However, the real-world utility of such models is limited by their high computational requirements, which makes them impractical for resource-constrained environments such as primary care facilities and conflict zones. Furthermore, shifts in the imaging domain can render these models ineffective and even compromise patient safety if such errors go undetected. To address these challenges, we propose M3D-NCA, a novel methodology that leverages Neural Cellular Automata (NCA) segmentation for 3D medical images using n-level patchification. Moreover, we exploit the variance in M3D-NCA to develop a novel quality metric which can automatically detect errors in the segmentation process of NCAs. M3D-NCA outperforms the two magnitudes larger UNet models in hippocampus and prostate segmentation by 2% Dice and can be run on a Raspberry Pi 4 Model B (2GB RAM). This highlights the potential of M3D-NCA as an effective and efficient alternative for medical image segmentation in resource-constrained environments.

<div>
<img src="/src/images/model_M3DNCA.png" width="600"/>
</div>



To get started with this repository simply follow these few steps:

## Quickstart

1. Install requirements of repository: `pip install -r requirements.txt `
2. Download prostate dataset from: http://medicaldecathlon.com/
3. Adapt **img_path** and **label_path** in **train_M3D_NCA.ipynb**
4. Run **train_M3D_NCA.ipynb**
5. To view results in tensorboard: `tensorboard --logdir path`
