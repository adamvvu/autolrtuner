import math 
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as tfk
import matplotlib.pyplot as plt
from statistics import NormalDist
import warnings

class AutoLRTuner:
    """
    Heuristically optimizes the learning rate in neural networks through subsampling loss differentials
    """
    def __init__(self, model, seed=42, DTYPE=tf.float32):
        """
        Initialize the tuner with a pre-compiled TensorFlow Keras `model` 

        Note: The compiled loss function in the model must be a callable, for example
        `tf.keras.losses.MeanSquaredError()` as opposed to `'mse'` when compiling.
        """
        self.model = model
        self.DTYPE = DTYPE
        self.initialized = False
        self.initParams = {}
        self.results = defaultdict(list)
        tf.random.set_seed(seed)
        np.random.seed(seed)

    def Tune(self, X, Y, lr_space=(1e-8, 1), num_subsamples=100, num_evals=100, batch_size=32, **kwargs):
        """
        Runs the learning rate tuning algorithm

        Args:
            X, Y            (array-like)  Features, Outcomes
            lr_space        (tuple)       Minimum and maximum learning rates to search over
            num_subsamples  (int)         Number of subsamples
            num_evals       (int)         Number of grid evaluations in the search space
            batch_size      (int)         Size of each subsample
            **kwargs                      Optional arguments to pass into `model.fit`
        """
        X, Y = self._parse_data(X), self._parse_data(Y)
        if len(X) != len(Y):
            raise Exception('Shape mismatch.')
        idx = np.array([i for i in range(len(X))]).astype(int)
        
        # Initial base model if necessary
        if not self.initialized:
            x = X[0:1,:]
            self._initialize(x)
            self.numBatches = math.ceil(X.shape[0] / batch_size)

        # Learning rate search space
        lrParamSpace = np.linspace(lr_space[0], lr_space[1], num=num_evals)
        for n in tqdm(range(num_subsamples)):

            # Subsample data
            sub_idx = tf.constant(np.random.choice(idx, size=batch_size, replace=False), dtype=tf.int32)
            X_sub, Y_sub = tf.gather(X, sub_idx), tf.gather(Y, sub_idx)

            # Evaluate gradients
            with tf.GradientTape() as tape:
                baseLoss = self._compute_loss(X_sub, Y_sub)
            grads = tape.gradient(baseLoss, self.model.trainable_variables)

            # Estimate loss differential on grid
            for lr in lrParamSpace:

                # Single backpropagation step
                self._set_lr(lr)
                self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                # Loss differential
                lossPost = self._compute_loss(X_sub, Y_sub)
                lossDiff = lossPost - baseLoss
                lossDiff = tf.math.reduce_mean(lossDiff).numpy()
                self.results[str(lr)].append(lossDiff)

                self._restore_model()
        
    def Plot(self, alpha=0.05, trim=0.05, xlim=None):
        """
        Plots the estimated loss differentials

        Args:
            alpha  (float)  Percentile for confidence intervals
            trim   (float)  Trims the y-axis for visualization in case of large outliers
            xlim   (tuple)  Optional tuple of (x_min, x_max) for zooming into subset of learning rates
        """
        results = self.GetResults()
        idx = results.columns.values.astype(float)
        plt.figure()
        if xlim is not None:
            idx = idx[(idx <= xlim[-1]) & (idx >= xlim[0])]
            results = results.loc[:,idx.astype('str')]
            plt.xlim(xlim)
        
        # Means/CIs
        means = results.mean(axis=0)
        se = results.std(axis=0) / math.sqrt(results.shape[0])
        dist = NormalDist()
        z = abs(dist.inv_cdf(alpha/2))
        upper_ci = means + z*se
        lower_ci = means - z*se

        # Plots
        plt.plot(idx, means.values)
        plt.fill_between(idx, lower_ci.values, upper_ci.values, color='orange', alpha=0.3)
        plt.axhline(y=0, linestyle='--', color='black', alpha=0.7)
        yu, yl = upper_ci.quantile(1-trim), lower_ci.quantile(trim)
        plt.ylim((yl, yu))
        plt.xlabel('Learning Rate')
        plt.ylabel('Change in Loss')
        plt.title('Estimated Loss Differentials')
        plt.show()

    def GetBestLR(self, method='mean'):
        """
        Gets the "best" LR according to some criteria

        Options available are:
            - 'mean' : Learning rate that best minimizes the loss on average
            - 'median' : Learning rate that best minimizes the loss on average
            - 'variance' : Most stable learning rate with smallest variance across subsamples
            - 'max' : Largest learning rate where loss differentials are all negative. Potentially usable as an upper bound on optimal LR
        """
        results = self.GetResults()
        if method == 'mean':
            x = results.mean(axis=0)
            bestLR = float(x.idxmin())
        elif method == 'median':
            x = results.median(axis=0)
            bestLR = float(x.idxmin())
        elif method == 'variance':
            x = results.std(axis=0)
            bestLR = float(x.idxmin())
        elif method == 'max':
            nonNeg = results.columns[(results <= 0).all(axis=0)].astype(float)
            if not nonNeg.empty:
                bestLR = nonNeg.max()
            else:
                warnings.warn('All learning rates had some positive loss differentials.')
                bestLR = 0.
        else:
            raise Exception('Invalid method.')
        
        return bestLR

    def GetCyclicLRCallback(self, stepsize=4, t_mul=2, m_mul=0.9, alpha=1e-8):
        """
        Returns a pre-configured TensorFlow Keras for use in training with cosine annealing

        Other hyperparameters of the learning rate scheduler should be tuned as needed.
        """
        # Initial upper bound on LR
        maxLR = self.GetBestLR(method='max')
        if maxLR == 0.:
            lrs = []
            # Failsafe: Revert to max of other methods
            for method in ['mean', 'median', 'variance']:
                lr = self.GetBestLR(method=method)
                lrs.append(lr)
            maxLR = max(lrs)
        
        callback = tfk.optimizers.schedules.CosineDecayRestarts(
                        initial_learning_rate=maxLR,
                        first_decay_steps=stepsize*self.numBatches,
                        t_mul=t_mul,
                        m_mul=m_mul,
                        alpha=alpha
                )
        return callback

    def GetResults(self):
        """
        Returns the raw results in a DataFrame of shape (num_simulations, num_evaluations)
        """
        if not self.results:
            raise Exception('No results found. Run `.Tune` first.')
        return pd.DataFrame(self.results)

    def _set_lr(self, lr=0.01):
        """
        Updates the learning rate
        """
        self.model.optimizer.learning_rate = float(lr)

    def _compute_loss(self, x, y):
        """
        Evalutes the loss function
        """
        y_hat = self.model(x, training=True)
        loss = self.model.loss(y, y_hat)
        return loss
    
    def _parse_data(self, Z):
        """
        Parses data into numpy arrays of shape (n, m>=1)
        """
        if isinstance(Z, pd.DataFrame):
            Z = Z.values
        dims = len(Z.shape)
        if dims < 2:
            Z = np.expand_dims(Z, axis=-1)
        return tf.constant(Z, dtype=self.DTYPE)

    def _initialize(self, x):
        """
        Initializes model states
        """
        # Initialize NN weights if necessary
        self.model(x)

        # Store initial states
        self.initParams = {
            'W' : self.model.get_weights(),
            'lr' : self.model.optimizer.learning_rate,
            'WOpt' : self.model.optimizer.get_weights()
        }
        self.initialized = True
    
    def _restore_model(self):
        """
        Restore model to original state
        """
        if not self.initialized:
            raise Exception('Model not initialized.')
        try:
            self.model.optimizer.learning_rate = self.initParams['lr']
            self.model.optimizer.set_weights(self.initParams['WOpt'])
            self.model.set_weights(self.initParams['W'])
        except:
            pass