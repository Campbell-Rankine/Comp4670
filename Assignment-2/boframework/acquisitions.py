import numpy as np
import numpy.typing as npt
from scipy.stats import norm

# Functional Structure


def probability_improvement(X: npt.ArrayLike, X_sample: npt.ArrayLike,
                            gpr: object, xi: float = 0.01) -> np.ndarray:
    """
    Probability improvement acquisition function.

    Computes the PI at points X based on existing samples X_sample using
    a Gaussian process surrogate model

    Arguments:
    ----------
        X: array-like of shape (m, d)
            The point for which the expected improvement needs to be computed.

        X_sample: array-like of shape (n, d)
            Sample locations

        gpr: GPRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.

        xi: float. Default 0.01
            Exploitation-exploration trade-off parameter.

    Returns:
    --------
        PI: ndarray of shape (1,)
    """
    # TODO Q2.4
    # Implement the probability of improvement acquisition function

    # FIXME

    raise NotImplementedError


def expected_improvement(X: npt.ArrayLike, X_sample: npt.ArrayLike,
                         gpr: object, xi: float = 0.01) -> np.ndarray:
    """
    Expected improvement acquisition function.

    Computes the EI at points X based on existing samples X_sample using
    a Gaussian process surrogate model

    Arguments:
    ----------
        X: array-like of shape (m, d)
            The point for which the expected improvement needs to be computed.

        X_sample: array-like of shape (n, d)
            Sample locations

        gpr: GPRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.

        xi: float. Default 0.01
            Exploitation-exploration trade-off parameter.

    Returns:
    --------
        EI : ndarray of shape (1,)
    """

    # TODO Q2.4
    # Implement the expected improvement acquisition function

    # FIXME

    raise NotImplementedError
