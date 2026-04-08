# Breaking Through the Noisy Correspondence: A Robust Model for Image-Text Matching

 A robust cross-modal retrieval framework that effectively handles noisy image-text correspondence through similarity distribution modeling and calibrated similarity learning, achieving state-of-the-art performance on three benchmarks under various noise rates.

## Authors

Haitao Shi<sup>1</sup>, Meng Liu<sup>2</sup> \*, Xiaoxuan Mu<sup>1</sup>, Xuemeng Song<sup>1</sup>, Yupeng Hu<sup>1</sup>, Liqiang Nie<sup>3</sup> \*

<sup>1</sup> <Shandong University, School of Software, Jinan, China>  
<sup>2</sup> <Shandong Jianzhu University, School of Computer Science and Technology, Jinan, China>
<sup>3</sup> <Harbin Institute of Technology (Shenzhen), School of Computer Science and Technology, Shenzhen, China>
\* Corresponding author

## Links

- **Paper**: [Paper Link](https://dl.acm.org/doi/10.1145/3662732)

## Updates

- [04/2026] Initial release of code and documentation

## Introduction

We present **DSDMR**, a framework for **Image-Text Matching**.  
Our method addresses **`Noisy Correspondence`** by enhancing noise robustness through three major innovations:
*Similarity Distribution Modeling: Transforms noise filtering into a parameter estimation problem of a bimodal Gaussian Mixture Model (GMM), explicitly separating “clean” and “noisy” distributions.
DSDMR Loss Function: Dynamically adjusts the margin to enhance the separability of the two distributions and mitigate gradient misguidance from noisy samples.
*Plug-and-Play Framework: As a post-processing module, it can be seamlessly integrated into any pretrained cross-modal model (e.g., CLIP) without modifying the original architecture. 
This repository provides the official implementation and evaluation scripts.

## Framework

![Framework](./assets/framework.png)

**Figure 1.** Overall framework of `DSDMR`.

## Usage
### Training

```bash
python scripts/train.py
```

### Inference

```bash
python scripts/infer.py
```

### Evaluation

```bash
python scripts/eval.py
```

## Citation

```bibtex
@article{DSDMR,
  author       = {Haitao Shi and
                  Meng Liu and
                  Xiaoxuan Mu and
                  Xuemeng Song and
                  Yupeng Hu and
                  Liqiang Nie},
  title        = {Breaking Through the Noisy Correspondence: {A} Robust Model for Image-Text
                  Matching},
  journal      = {{ACM} Trans. Inf. Syst.},
  volume       = {42},
  number       = {6},
  pages        = {149:1--149:26},
  year         = {2024}
}
```
## Acknowledgement

- Thanks to our supervisor and collaborators for valuable support.
- Thanks to the open-source community for providing useful baselines and tools.

## License

This project is released under the Apache License 2.0.
