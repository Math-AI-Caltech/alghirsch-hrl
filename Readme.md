# HRL for Algebraic Hirsch Problem

Searches for linear square-free monomial ideals with diameter larger than the degree. This is a repo for the paper: [Hierarchical Reinforcement Learning for Sparse-Reward Search in Commutative Algebra (ICML 2026)](https://openreview.net/forum?id=DF6jVG4fG8):
* **ICML Project page**: https://icml.cc/virtual/2026/poster/65477
* **arXiv**: https://arxiv.org/abs/2606.22922

## Install
Requires Python ≥ 3.9. Additionally requires C++ toolchain to compile backend for syzygies and diameter. CUDA is optional.
```bash
pip install -r requirements.txt
pip install --no-build-isolation ./structural/diameter
pip install --no-build-isolation ./structural/syzygies/cuda
pip install -e .
```

## Usage
```bash
python3 train.py
python3 train_dual.py
```

## Citation
```bibtex
@inproceedings{butbaia2026hierarchical,
  author    = {Giorgi Butbaia and Paul Orland and Coco Huang and Davide Passaro and Lucas Fagan and Michele Tarquini and Hailong Dao and David Eisenbud and Ali Shehper and Sergei Gukov},
  title     = {Hierarchical Reinforcement Learning for Sparse-Reward Search in Commutative Algebra},
  booktitle = {Proceedings of the Forty-Third International Conference on Machine Learning},
  year      = {2026},
  url       = {https://openreview.net/forum?id=DF6jVG4fG8}
}
```