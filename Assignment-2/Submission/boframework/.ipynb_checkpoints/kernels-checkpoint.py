import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from boframework.utils import Kernel, Hyperparameter

# Class Structure


class Matern(Kernel):
    """
    Matern kernel.

    Arguments:
    ----------
    length_scale : float
        The length scale of the kernel. 

    length_scale_bounds : pair of floats >= 0, default=(1e-5, 1e3)
        The lower and upper bound on 'length_scale'.

    variance : float
        The signal variance of the kernel

    variance_bounds : pair of floats >= 0, default=(1e-5, 1e2)
        The lower and upper bound on 'variance'.

    nu : float, default=1.5
        The parameter nu controlling the smoothness of the learned function.
    """

    def __init__(self, length_scale: float = 1.0, length_scale_bounds: tuple = (1e-5, 1e3),
                 variance: float = 1.0, variance_bounds: tuple = (1e-5, 1e2),
                 nu: float = 1.5) -> None:
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.variance = variance
        self.variance_bounds = variance_bounds
        self.nu = nu

    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    @property
    def hyperparameter_variance(self):
        return Hyperparameter("variance", "numeric", self.variance_bounds)

    def __call__(self, X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
        """
        Return the kernel k(X, Y).

        Arguments:
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            should be evaluated instead.

        Returns:
        --------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        """

        X = np.atleast_2d(X)
        length_scale = np.squeeze(self.length_scale).astype(float)

        # TODO Q2.1
        # Implement the Matern class covariance functions for the Matern32 and Matern52 cases
        
        # HINT: To compute the pairwise Euclidean distances, we could research how to use pdist and cist
        #       (check: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html,
        #               https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)
        #       or come up with your own solution using numpy. 
        if not Y is None:
            d = cdist(X / length_scale, Y / length_scale, metric="euclidean") #Scale inside the distance measurement for clarity of code later on
        else:
            d = pdist(X / length_scale, metric="euclidean")
        if self.nu == 1.5:
            # FIXME
            #Matern32 implementation
            #if length_scale > self.length_scale_bounds[0] and length_scale < self.length_scale
            toret =self.variance * (1.+ np.sqrt(3.)*d) * np.exp(-np.sqrt(3.)*d)
        elif self.nu == 2.5:
            # FIXME
            #Matern52 implementation
            inner = np.sqrt(5)*d
            toret = self.variance*(1.+inner+inner**2 / 3.) * np.exp(-inner)
            pass
        else:
            # Do not change
            raise NotImplementedError
        if Y is None: #Must squareform the kernel and force positive semidefinite matrix
            toret = squareform(toret)
            np.fill_diagonal(toret, 1)
        return toret
