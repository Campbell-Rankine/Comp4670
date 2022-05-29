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

    stdX = np.std(X)
    if stdX == 0:
        return 0
    m_gp = gpr.predict(X_sample)
    Z = np.mean(X, axis = 1) - np.mean(m_gp) -xi
    Z = Z/(np.std(X))
    return norm.cdf(Z)
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
    stdX = np.std(X)
    if stdX == 0:
        return 0
    else:
        m_gp = gpr.predict(X_sample)
        Zi = np.mean(X, axis=1) - np.mean(m_gp) -xi
        Z = Zi/np.std(X) #now have z
        return (Zi*norm.cdf(Z) + stdX*norm.pdf(Z))
        
    raise NotImplementedError
