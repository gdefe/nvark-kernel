# Time Series Kernels based on Nonlinear Vector AutoRegressive Delay Embeddings (NeurIPS 2023)

[![NIPS](https://img.shields.io/badge/NIPS-2023-blue.svg?style=flat-square)]()

Code implementation and official repository for the paper "Time Series Kernels based on Nonlinear Vector AutoRegressive Delay Embeddings" (NeurIPS 2023)

**Authors**: [Giovanni De Felice](mailto:gdefe@liverpool.ac.uk), Yannis Goulermas, Vladimir Gusev

---

## TLDR

We propose NVARk, a novel kernel for univariate and multivariate time series by integrating NVAR-made dynamics into reservoir-based kernel architectures.
NVARk compares time series based on the linear dynamics of NVAR embeddings, which are built from concatenating lags and nonlinear functionals to the original series.
In terms of accuracy, NVARk outperforms the corresponding RC architecture.
Computationally, it is exceptionally efficient and based on a few integer hyperparameters, which together allow for further improvement of the results with simple supervised grid-based optimization.

---

## Requirements

We run all the experiments in `python 3.9`. To solve all dependencies, we recommend using Anaconda and creating a new environment.

```
conda env create -f conda_env.yml
conda activate nvarkernel_test
```

For a quick example, run 'python main.py'. This executes an SVM classification on the JapaneseVowels dataset with the NVARk general setting.
More settings / experiments / datasets are available inside the code, just uncomment the desired section.

## Bibtex reference

If you find this code useful please consider citing our paper:

```
@inproceedings{
felice2023time,
title={Time Series Kernels based on Nonlinear Vector AutoRegressive Delay Embeddings},
author={Giovanni De Felice and John Y Goulermas and Vladimir Gusev},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=UBUWFEwn7p}
}
```
