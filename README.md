# Robustness of Spiking Neural Networks under Noise: MSc AI & CS Final Project

This repository contains the final project for the **MSc in Artificial Intelligence & Computer Science**.  
We study training and robustness of Spiking Neural Networks (SNNs) on **MNIST** using **Poisson encoding**, and evaluate how additive noise degrades performance under several testing regimes.

## TL;DR
- **Data**: MNIST digits, encoded as **Poisson spike trains**.
- **Noise**: additive Gaussian noise at multiple SNR levels (dB).
- **Tasks**: digit classification under clean and noisy conditions.
- **Comparisons**: three test pipelines + two noise-training protocols.

## Datasets & Encoding
- **Dataset**: MNIST (28×28 grayscale).
- **Encoding**: Each pixel intensity is converted to a **Poisson spike train** whose firing rate is proportional to the pixel value over a fixed window. This yields event-based inputs suitable for SNN simulation.

## Test Pipelines
1. **SNN with STDP training + Random Forest**
   - The SNN is trained using **STDP** (unsupervised synaptic plasticity).
   - Spiking activity (e.g., spike counts / rates / layer-wise maps) is extracted as features.
   - A **RandomForest** classifier is trained on these SNN-derived features.

2. **SNN without STDP training + Random Forest**
   - The SNN is **frozen** (STDP off).  
   - We still pass data through the network and extract spiking features.
   - A **RandomForest** is trained on these fixed features.

3. **Random Forest only**
   - A baseline that trains **RandomForest directly on non-spiking features** (e.g., raw pixels or simple transforms) without an SNN front-end.

## Noise Training Protocols (Experiments)
We investigate two complementary setups:

1. **Train on noisy data → Test on noisy data**  
   Models are exposed to noise during training to potentially learn noise-invariant decision boundaries, then evaluated at matched SNRs.

2. **Train on noisy data → Test on clean data**  
   Models trained with noise are evaluated on clean inputs to assess potential **regularization** or **over-robustness** effects (e.g., do they still perform well when noise disappears?).

> For reproducibility, seeds are fixed where applicable, and intensity clipping is applied after noise addition to keep values in a valid range.

## Feature Extraction (SNN → ML)
- Inputs (Poisson spike trains) pass through the SNN.  
- We extract **handy summary statistics** from spiking activity, e.g.:
  - Spike counts per neuron/filter over a time window,
  - Average firing rates,
  - (Optionally) simple spatial pooling of spike maps.
- These fixed-length vectors serve as features for **RandomForest**.

## Why can RandomForest outperform SNN-based pipelines?
- **Information loss in spiking features**: collapsing rich spatiotemporal dynamics to aggregate counts/rates can discard discriminative detail that **RF on raw pixels** still sees.
- **Optimization maturity**: RF on tabular or pixel features is a very strong classical baseline with few hyperparameters, good bias–variance trade-off (bagging), and often excellent default performance.
- **Capacity and training stability**: SNNs can be **sensitive to encoding choices, timing, and STDP hyperparameters**; suboptimal settings or short training windows may underutilize the data compared to RF.

## Why might SNN accuracy degrade faster under noise?
- **Temporal sensitivity**: Poisson rate coding and spiking thresholds make the effective signal more fragile—noise perturbs spike timings and counts, shifting neurons across firing thresholds.
- **Cumulative distortions**: Early-layer spike perturbations propagate, altering downstream activity more than additive noise alters raw pixel intensities.
- **Unsupervised plasticity mismatch**: If STDP learned on clean data, its learned filters may be less aligned with noisy statistics, causing larger drops versus a robust, bagged RF.
