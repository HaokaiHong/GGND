# Geometric Graph Neural Diffusion (GGND) for Stable Molecular Dynamics Simulations

Official implementation of **Geometric Graph Neural Diffusion (GGND)** integrated with **MACE** for learning **generalizable molecular force fields**.

---

## 🧠 Overview

Molecular dynamics simulations rely on accurate force fields, yet even small prediction errors can accumulate and destabilize long-time trajectories when molecular conformations deviate from the training distribution. 

This repository implements **Geometric Graph Neural Diffusion (GGND)**, a plug-and-play module that enhances equivariant graph neural networks with a global diffusion mechanism. GGND models feature evolution through an equivariant diffusion process on fully connected molecular graphs, capturing geometrically invariant topological structures and improving robustness to conformational shifts. 

By mitigating error accumulation under distributional shifts, GGND improves both predictive accuracy and stability in real molecular dynamics simulations. 

---

## ✨ Highlights

- Stable Molecular Dynamics via Diffusion on Geometric Graphs
- Robustness to Geometric Topological Shifts
- Theoretical Guarantees for Generalization
- Plug-and-Play Integration with Equivariant GNNs
- Empirical Validation in Real MD Simulations

---

## 📊 Datasets

Download and place the datasets in:

```
./datasets/
```

### 3BPA

* Source: [https://pubs.acs.org/doi/10.1021/acs.jctc.1c00647](https://pubs.acs.org/doi/10.1021/acs.jctc.1c00647)
* Usage: Default benchmark for force field learning and stability evaluation

### SAMD23

* Source: [https://github.com/SAITPublic/MLFF-Framework](https://github.com/SAITPublic/MLFF-Framework)
* Usage: Cross-dataset generalization evaluation

Ensure all files are extracted and formatted according to the preprocessing scripts.

---

## ⚙️ Environment Setup

This project builds upon the **MACE** framework.

Follow installation instructions from:
[https://github.com/ACEsuit/mace](https://github.com/ACEsuit/mace)

### Requirements

* Python ≥ 3.10
* PyTorch (CUDA recommended)
* NumPy
* ASE
* Other dependencies listed in the MACE repository

Example setup:

```bash
conda create -n ggnd python=3.10
conda activate ggnd
pip install -r requirements.txt
```

---

## 🚀 Training

### Train on 3BPA

```bash
sh sh/train-3BPA.sh
```

---

## 📈 Evaluation

After training, run evaluation scripts to perform MD and analyze stability:

Example:

```bash
sh evaluation/run_md.sh
```
---

## 📖 Citation

If you find this work useful, please cite:

```
@inproceedings{hong2026ggnd,
  title={Geometric Graph Neural Diffusion for Stable Molecular Dynamics Simulations},
  author={Hong, Haokai and Lin, Wanyu and Zhang, Chusong and Tan, Kay Chen},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

---

## 🤝 Acknowledgements

This implementation builds upon the MACE framework and related open-source molecular simulation tools.