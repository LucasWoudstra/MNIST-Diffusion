# MNIST Diffusion: Timestep Sampling Bias Experiments

This repository contains the source code and experimental pipeline for training and evaluating a Denoising Diffusion Probabilistic Model (DDPM) on the MNIST dataset. The project explores the performance impact of shifting the training focus toward different noise levels by applying custom non-uniform probability distributions during the timestep ($t$) sampling phase.

The core implementation modifications reside across two principal modules:

- **`model_experimental.py`** — Contains the custom probability-weighted timestep selection architecture within the forward diffusion pass.
- **`train_mnist_new.py`** — Houses the automated training loop, the tracking of evaluation metrics, and the command-line execution framework.

---

## Requirements & Environment Setup

This implementation requires **Python 3.8+** and a **CUDA-capable GPU** for efficient training and evaluation.

1. Clone or extract this repository to your local machine.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Experimental Setup

The framework supports five distinct timestep ($t$) sampling strategies via the `--experiment` flag, implemented inside `model_experimental.py`:

| Experiment Flag | Sampling Distribution Description        | High-Level Focus                               |
|-----------------|------------------------------------------|------------------------------------------------|
| `uniform`       | Standard uniform distribution            | Balanced focus across all noise levels         |
| `A`             | Linear ramp increasing from $0$ to $T-1$ | Focuses heavily on high noise (late timesteps) |
| `B`             | Linear ramp decreasing from $0$ to $T-1$ | Focuses heavily on low noise (early timesteps) |
| `C`             | Inverted U-shape (centered peak)         | Focuses on intermediate noise levels           |
| `D`             | U-shape (peaks at boundaries)            | Focuses on extreme high and low noise levels   |
| `all`           | Sequential Execution Routine             | Runs experiments A, B, C, and D sequentially   |

---

## How to Run the Experiments

All configurations are managed directly from the command line using `train_mnist_new.py`.

### 1. Running a Single Configuration

To train the model using a specific experiment distribution (e.g., Experiment C) with a cosine noise schedule for 40 epochs:

```bash
python train_mnist_new.py --experiment C --noise_schedule cosine --epochs 40 --batch_size 128
```

### 2. Automated Batch Execution (Replicating All Experiments)

To replicate the entire experimental suite consecutively under identical hyperparameters:

```bash
python train_mnist_new.py --experiment all --noise_schedule cosine --epochs 40
```

### 3. Primary Command-Line Arguments

Customize execution by overriding default hyperparameters with these flags:

| Flag                  | Description                                                           | Default   |
|-----------------------|-----------------------------------------------------------------------|-----------|
| `--experiment`        | Timestep sampling distribution (`uniform`, `A`, `B`, `C`, `D`, `all`) | `uniform` |
| `--noise_schedule`    | Variance schedule (`cosine` or `linear`)                              | `cosine`  |
| `--epochs`            | Total training cycles                                                 | `40`      |
| `--lr`                | Peak learning rate for the OneCycleLR scheduler                       | `0.001`   |
| `--batch_size`        | Batch size for training                                               | `128`     |
| `--eval_freq`         | Interval of epochs between intermediate validation stages             | `8`       |
| `--fid_eval_samples`  | Number of samples for fast intermediate FID calculation               | `1000`    |
| `--fid_final_samples` | Number of samples for rigorous final FID benchmarking                 | `10000`   |
| `--no_clip`           | Enable normal sampling without clipping $x_0$ to $[-1.0, 1.0]$        |           |
| `--cpu`               | Force CPU training instead of CUDA                                    |           |

---

## Project Outputs & Evaluation Metrics

When an experiment is launched, the script automatically isolates all outputs in a dedicated directory:

```
results/experiment_<FLAG>_<EPOCHS>_<SCHEDULE>/
```

Each folder will contain:

- **Model Checkpoints (`.pt`)** — Saved weights captured automatically at your specified `--eval_freq` and upon pipeline completion.
- **Generated Image Grids (`.png`)** — Visual validation samples produced by the Exponential Moving Average (EMA) model at evaluation intervals.
- **Quantitative Metrics (`intermediate_fid_scores.csv`)**
  - *Intermediate Validation*: Tracks progressive model performance using a fast 1,000-sample Fréchet Inception Distance (FID) score at evaluation intervals.
  - *Final Benchmarking*: Logs a rigorous, publication-ready 10,000-sample FID score calculated immediately after the final epoch completes.
