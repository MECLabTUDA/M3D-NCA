<div>
<img src="/src/images/MED_NCA_header.png" style="width: 100%;"/>
</div>

# MED-NCA: Robust and Lightweight Segmentation with Neural Cellular Automata

This repository contains the offical framework of the two award winning Neural Cellular Automata based segmentation methods Med-NCA and M3D-NCA. Within a minute you can train your own NCA segmentation task. 

<div>
<p align="center">
<img src="/src/images/Med_NCA_introduction.png" width="70%"/>
</p>
</div>

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Method](#method)
4. [Quality Metric](#quality-metric)
5. [Result Tracking]()
6. [Citations](#citations)
7. [Contact](#contact)

## Introduction
> Medical image segmentation relies heavily on large-scale deep learning models, such as UNet-based architectures. However, the real-world utility of such models is limited by their high computational requirements, which makes them impractical for resource-constrained environments such as primary care facilities and conflict zones. 

Neural Cellular Automata are locally communicating models that due to their one-cell model size have only 13k parameters and require merely 50kB of storage. By iterating this learnt rule over each cell of an image advanced tasks such as medical image segmentation can be achieved. 

The difficulty when training NCAs on high-resolution images is the required number of steps to gather global knowledge aswell as the VRAM requirements. 

This framework solves these issues and is ready to be applied to medical image segmentation tasks. [Try it out yourself!](#installation)

<div>
<p align="center">
<img src="/src/images/M3D_Inference.gif" width="40%"/>
</p>
</div>

## Installation
**To get start with this repository follow these steps:**

1. Install requirements of repository: 
    ```
    pip install -r requirements.txt
    ```
2. Download hippocampus or prostate dataset from: 
    ```
    http://medicaldecathlon.com/
    ```
3. Adapt **img_path** and **label_path** in **train_M3D_NCA.ipynb**
4. Run **train_M3D_NCA.ipynb**
5. To view results in tensorboard: 
    ```
    tensorboard --logdir path
    ```

## Method
MED-NCA uses a multi level architecture to deal with the issue of global communication. During training patches are used on the high-res levels of the architecture to drastically reduce the required VRAM.
### Architecture 
<div>
<p align="center">
<img src="/src/images/model_M3DNCA.png" width="50%"/>
</p>
</div>


## Quality Metric
The random activation of NCAs results in slightly different results, when MED-NCA is run multiple times on the same image. By observing the variance inbetween predictions we can make an estimate of the quality of the predicted segmentation.

<div>
<p align="center">
<img src="/src/images/qm.png" width="50%"/>
</p>
</div>

## Result Tracking

### Visualize prediction and variance

To enable the visual outputsin the Jupyter Notebooks run:
```
agent.getAverageDiceScore(pseudo_ensemble=True, showResults=True)
```
Per Patient this will output a combination of three images, where the first one shows the middle slice of the image 3D volume. The second image shows the predicted segmentation. Lastly the third image shows the variance map.

The output will then look the following:
<div>
<p align="center">
<img src="/src/images/Result_DiceVariance.png" width="50%"/>
</p>
</div>


### Track experiments with tensorboard
To view results in tensorboard and track the loss simply run the following command: 
    ```
    tensorboard --logdir path
    ```

<div>
<p align="center">
<img src="/src/images/TensorboardOutputs.png" width="50%"/>
</p>
</div>

## Citations

If you are using this framework, please cite the according paper.  

### [Med-NCA](https://arxiv.org/pdf/2302.03473.pdf)
```
@inproceedings{kalkhof2023med,
  title={Med-NCA: Robust and Lightweight Segmentation with Neural Cellular Automata},
  author={Kalkhof, John and Gonz{\'a}lez, Camila and Mukhopadhyay, Anirban},
  booktitle={International Conference on Information Processing in Medical Imaging},
  pages={705--716},
  year={2023},
  organization={Springer}
}
```
### [M3D-NCA](https://arxiv.org/pdf/2309.02954.pdf)
```
@inproceedings{kalkhof2023m3d,
  title={M3D-NCA: Robust 3D Segmentation with Built-In Quality Control},
  author={Kalkhof, John and Mukhopadhyay, Anirban},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={169--178},
  year={2023},
  organization={Springer}
}
```

### Contact
If you have any questions or feedback regarding the repository feel free to contact me via [Twitter/X](https://twitter.com/kalkjo) or [Linkedin](https://www.linkedin.com/in/john-kalkhof-708639261/).



