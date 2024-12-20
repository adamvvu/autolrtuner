import pytest
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk

from autolrtuner.autolrtuner import AutoLRTuner

np.random.seed(42)

def _GenerateData(N=100, numFeatures=3):
    X = np.random.normal(size=(N,numFeatures))
    eps = np.random.normal(size=N, scale=0.1)
    Y = np.sum(np.log(np.abs(X)+1), axis=-1) + eps
    return X, Y

def _Compile_dnn(numFeatures, architecture=[32,32]):
    inputLayer = tfk.layers.Input(shape=(numFeatures,))
    x = inputLayer
    for nodes in architecture:
        x = tfk.layers.Dense(nodes, activation='relu')(x)
    x = tfk.layers.Dense(1)(x)

    model = tfk.Model(inputs=inputLayer, outputs=x)
    model.compile(optimizer=tfk.optimizers.Adam(), 
                  loss=tfk.losses.MeanSquaredError())
    return model

def test_all():

    num_subsamples = 7
    num_evals = 9

    # Data
    X, Y = _GenerateData(100)

    # Compile model
    numFeatures = X.shape[-1]
    model = _Compile_dnn(numFeatures, architecture=[2])

    # Tune
    lrTuner = AutoLRTuner(model)
    lrTuner.Tune(X, Y, num_subsamples=num_subsamples, num_evals=num_evals, batch_size=10)

    ## Unit tests

    # Test: Results
    results = lrTuner.GetResults()
    assert results.shape == (num_subsamples, num_evals), 'Results shape mismatch'

    # Test: Plots
    lrTuner.Plot()
    lrTuner.Plot(xlim=(0,0.5))

    # Test: Estimates
    methods = ['mean', 'median', 'variance']
    for method in methods:
        x = lrTuner.GetBestLR(method=method)
        assert isinstance(x, float), f'Outputs invalid. Received {x}'

    return

if __name__ == '__main__':
    test_all()