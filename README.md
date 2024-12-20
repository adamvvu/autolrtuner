[![Build](https://img.shields.io/github/actions/workflow/status/adamvvu/autolrtuner/autolrtuner_tests.yml?style=for-the-badge)](https://github.com/adamvvu/autolrtuner/actions/workflows/autolrtuner_tests.yml)
[![PyPi](https://img.shields.io/pypi/v/autolrtuner?style=for-the-badge)](https://pypi.org/project/autolrtuner/)
[![Downloads](https://img.shields.io/pypi/dm/autolrtuner?style=for-the-badge)](https://pypi.org/project/autolrtuner/)
[![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](https://github.com/adamvvu/autolrtuner/blob/master/LICENSE)

Heuristically optimize learning rates in neural networks through subsampling loss differentials.

---

## Auto LR Tuner

The learning rate is often one of the most important hyperparameters when training neural networks. Adaptive gradient-based methods (e.g. ADAM), decreasing learning rates based on a validation set, and cosine annealing are common tricks done in practice to improve convergence.

This library provides a simple algorithm for automatically tuning learning rates for TensorFlow Keras models. Importantly, these methods are largely based on heuristics and my own experience training neural networks and there are no formal results.

### Algorithm Details

The main idea behind this implementation is to estimate the optimal learning rate by trying to determine the steepness of the loss surface. Intuitively, a very small learning rate leads to almost no change in losses, while an excessively large learning rate can overshoot local minima and even increase the loss.

We start with a user-specified grid $\Theta = [lr_{min}, lr_{max}]$ of potential learning rates to search over. Next, we subsample $m$ observations from the data, and evaluate $\Delta L := L_{post} - L_{pre}$ where $L_{pre}$ is the baseline loss value, and $L_{post}$ is the loss after a single backpropagation step for a given learning rate $\theta \in \Theta$. The subsampling process is repeated $M$ times to approximate $E_{M}[\Delta L]$ and construct confidence intervals based on the subsampled distribution $\hat{P}_{\Delta L, M}$.

In this interpretation of the problem, the "optimal" learning rate is the one that consistently decreases the loss across the dataset without excessively high variance. For example in the plot below, a small learning rate near-zero may be very slow to converge while a learning rate of one actually increases the loss on average. Here the optimal rate may be something close to 0.125. 

![Plot](assets/losses.png)

### Getting Started

Install from PyPi:

`$ pip install autolrtuner`

#### Dependencies

- `python >= 3.8`
- `tensorflow >= 2.5`
- `numpy`
- `pandas`
- `scipy`
- `tqdm`
- `matplotlib` *(Optional)*

#### Example

```
from autolrtuner import AutoLRTuner

# Compiled TensorFlow Keras model
model 

# Run the tuning algorithm
lrTuner = AutoLRTuner(model)
lrTuner.Tune(X, Y, num_subsamples=100, num_evals=100, batch_size=32)

# Get the optimal learning rate
bestLR = lrTuner.GetBestLR(method='mean')

# Plot the loss differentials
lrTuner.Plot()
```