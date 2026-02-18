# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the course materials repository for **COMP5329 - Deep Learning** (Semester 1, 2026). It contains weekly lecture materials (PDFs and lecture scripts) and tutorial Jupyter notebooks with hands-on PyTorch implementations.

## Working with the Materials

**Running notebooks:**
```bash
jupyter notebook
# or
jupyter lab
```

**Core dependencies:** PyTorch, NumPy, Matplotlib. Install via:
```bash
pip install torch torchvision numpy matplotlib
# See https://pytorch.org/get-started/locally/ for platform-specific PyTorch installs
```

No build system, test suite, or package configuration exists — this is a pure teaching materials repository.

## Structure

```
Lecture/      # PDF slides + detailed lecture scripts (.txt) per week
Tutorial/     # Jupyter notebooks (.ipynb) with PyTorch code per week
```

**Note:** Tutorial folder names and notebook filenames are offset by one week (e.g., `Tutorial/Week5 - Convolutional Neural Networks/` contains `Week6_CNN.ipynb`). The folder name reflects the topic, the filename reflects the lecture week number.

## Course Progression

| Week | Lecture Topic | Tutorial Content |
|------|--------------|-----------------|
| 1 | Introduction | — |
| 2 | Foundations of Deep Neural Networks | Tensor ops, Dataset, DataLoader |
| 3 | Regularization for Deep Models | MLP implementation, Optimizers |
| 4 | Convolutional Neural Networks | CNN basics, CNN architectures |
| 5 | Graph Neural Networks | Recurrent Neural Networks |

## Code Patterns in Tutorials

- Custom `Dataset` subclasses using `torch.utils.data.Dataset`
- `DataLoader` for batching
- NumPy comparisons shown alongside PyTorch equivalents
- Matplotlib for visualization
- Gaussian-distributed synthetic data for classification demos
