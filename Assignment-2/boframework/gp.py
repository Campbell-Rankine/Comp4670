from __future__ import annotations
from boframework.kernels import Matern
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize
import numpy.typing as npt
from typing import Sequence, Tuple, Union
import copy
from operator import itemgetter
import numpy as np

# Class Structure


class GPRegressor:
    """
    Gaussian process regression (GPR).

    Arguments:
    ----------
    kernel : kernel instance,
        The kernel specifying the covariance function of the GP.

    noise_level : float , default=1e-10
        Value added to the diagonal of the kernel matrix during fitting.
        It can be interpreted as the variance of additional Gaussian
        measurement noise on the training observations.

    n_restarts : int, default=0
        The number of restarts of the optimizer for finding the kernel's
        parameters which maximize the log-marginal likelihood. The first run
        of the optimizer is performed from the kernel's initial parameters,
        the remaining ones (if any) from thetas sampled log-uniform randomly
        (for more details: https://en.wikipedia.org/wiki/Reciprocal_distribution)
        from the space of allowed theta-values. If greater than 0, all bounds
        must be finite. Note that `n_restarts == 0` implies that one
        run is performed.

    random_state : int, RandomState instance
    """

    def __init__(self,
                 kernel: Matern,
                 noise_level: float = 1e-10,
                 n_restarts: int = 0,
                 random_state: int = np.random.RandomState(4)
                 ) -> None:

        self.kernel = kernel
        self.noise_level = noise_level
        self.n_restarts = n_restarts
        self.random_state = random_state

    def optimisation(self,
                     obj_func: callable,
                     initial_theta: npt.ArrayLike,
                     bounds: Sequence
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Function that performs Quasi-Newton optimisation using L-BFGS-B algorithm.

        Note that we should frame the problem as a minimisation despite trying to
        maximise the log marginal likelihood.

        Arguments:
        ----------
        obj_func : the function to optimise as a callable
        initial_theta : the initial theta parameters, use under x0
        bounds : the bounds of the optimisation search

        Returns:
        --------
        theta_opt : the best solution x*
        func_min : the value at the best solution x, i.e, p*
        """
        # TODO Q2.2
        # Implement an L-BFGS-B optimisation algorithm using scipy.minimize built-in function
        # FIXME
        #Assert initial value in bounds
        ORObj = minimize(obj_func, initial_theta, method="L-BFGS-B", bounds=bounds)
        return (ORObj.__getitem__(x), obj_func(ORObj.__getitem__(x))) #Return best result, and final min value
        raise NotImplementedError

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> GPRegressor:
        """
        Fit Gaussian process regression model.

        Arguments:
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature vectors or other representations of training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns:
        --------
        self : object
            The current GPRegressor class instance.
        """
        # TODO Q2.2
        # Fit the Gaussian process by performing hyper-parameter optimisation
        # using the log marginal likelihood solution. To maximise the marginal
        # likelihood, you should use the `optimisation` function

        # HINT I: You should run the optimisation (n_restarts) time for optimum results.

        # HINT II: We have given you a data structure for all hyper-parameters under the variable `theta`,
        #           coming from the Matern class. You can assume by optimising `theta` you are optimising
        #           all the hyper-parameters.

        # HINT III: Implementation detail - Note that theta contains the log-transformed hyperparameters
        #               of the kernel, so now we are operating on a log-space. So your sampling distribution
        #               should be uniform.

        self._kernel = copy.deepcopy(self.kernel)

        self._X_train = X
        self._y_train = y
        #Ky = self._kernel(self._X_train, self._X_train)
        # FIXME
        #Find theta0
        theta0 = self._kernel.theta
        lsb = self._kernel.length_scale_bounds
        vb = self._kernel.variance_bounds
        theta = [self._kernel.length_scale, self._kernel.variance]
        for i in range(self.n_restarts):
            #start by sampling theta from the bounds
            optim = optimisation(log_marginal_likelihood, theta, [lsb, vb])
            if optim[1] < minpair[1]:
                minpair = optim
        self.kernel.theta = optim[0]
        return self
        raise NotImplementedError

    def predict(self, X: npt.ArrayLike, return_std: bool = False) -> Union[np.ndarray, Tuple(np.ndarray, np.ndarray)]:
        """
        Predict using the Gaussian process regression model.

        In addition to the mean of the predictive distribution, optionally also
        returns its standard deviation (`return_std=True`).

        Arguments:
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points where the GP is evaluated.
        return_std : bool, default=False
            If True, the standard-deviation of the predictive distribution at
            the query points is returned along with the mean.

        Returns (depending on the case):
        --------------------------------
        y_mean : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Mean of predictive distribution a query points.
        y_std : ndarray of shape (n_samples,) or (n_samples, n_targets), optional
            Standard deviation of predictive distribution at query points.
            Only returned when `return_std` is True.
        """
        # TODO Q2.2
        # Implement the predictive distribution of the Gaussian Process Regression
        # by using the Algorithm (1) from the assignment sheet.
        self._kernel = copy.deepcopy(self.kernel)
        # FIXME
        k = self._kernel
        Ky = k(self._X_train, self._X_train) + self.noise_level*np.identity(len(k(self._X_train, self._X_train)))
        
        L = cholesky(Ky)
        alpha = cho_solve((L, False), self._y_train)
        ktstar = k(X, self._X_train)
        fstar = ktstar@alpha
        if return_std:
            kstart= k(self._X_train, X)
            v = solve_triangular(L, kstart, trans=1)
            
            #Solve this for v from algorithm 1
            Var=k(X, X) - (v.T @ v)
            return(fstar, np.sqrt(np.diag(Var))) #Return tuple
        else:
            return fstar #Else return singular
        raise NotImplementedError

    def fit_and_predict(self, X_train: npt.ArrayLike, y_train: npt.ArrayLike, X_test: npt.ArrayLike,
                        return_std: bool = False, optimise_fit: bool = False
                        ) -> Union[
        Tuple(dict, Union[np.ndarray, Tuple(np.ndarray, np.ndarray)]),
        Union[np.ndarray, Tuple(np.ndarray, np.ndarray)]
    ]:
        """
        Predict and/or fit using the Gaussian process regression model.

        Based on the value of optimise_fit, we either perform predictions with or without fitting the GP first.

        Arguments:
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Feature vectors or other representations of training data.
        y_train : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.
        X_test : array-like of shape (n_samples, n_features)
            Query points where the GP is evaluated.
        return_std : bool, default=False
            If True, the standard-deviation of the predictive distribution at
            the query points is returned along with the mean.
        optimise_fit : bool, default=False
            If True, we first perform fitting and then we continue with the
            prediction. Otherwise, perform only inference.

        Returns (depending on the case):
        --------------------------------
        kernel_params: A dictionary of the kernel (hyper)parameters, optional.
            Only for `optimise_fit=True` case;
            HINT: use `get_params()` fuction from kernel object.
        y_mean : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Mean of predictive distribution a query points.
        y_std : ndarray of shape (n_samples,) or (n_samples, n_targets), optional
            Standard deviation of predictive distribution at query points.
            Only returned when `return_std` is True.

        """
        # TODO Q2.6a
        # Implement a fit and predict or only predict scenarios. The course of action
        # should be chosen based on the variable `optimise_fit`.
        if optimise_fit:
            # FIXME
            raise NotImplementedError
        else:
            self._kernel = copy.deepcopy(self.kernel)

            self._X_train = X_train
            self._y_train = y_train

            # FIXME
            raise NotImplementedError

    def log_marginal_likelihood(self, theta: npt.ArrayLike) -> float:
        """
        Return log-marginal likelihood of theta for training data.

        Arguments:
        ----------
        theta : array-like of shape (n_kernel_params,)
            Kernel hyperparameters for which the log-marginal likelihood is
            evaluated.

        Returns:
        --------
        log_likelihood : float
            Log-marginal likelihood of theta for training data.
        """
        # TODO Q2.2
        # Compute the log marginal likelihood by using the Algorithm (1)
        # from the assignment sheet.

        kernel = self._kernel
        kernel.theta = theta

        # FIXME
        Ky = kernel(self._X_train, self._X_train) + self.noise_level * np.identity(len(kernel(self._X_train, self._X_train)))
        L = cholesky(Ky)
        alpha = cho_solve((L, False), self._y_train)
        print("kernal y: " + str(Ky), "L: " + str(L), "alpha: " + str(alpha))
        ret = (-0.5*self._y_train.T @ alpha) - np.sum(np.log(np.diag(L))) - (len(self._y_train))/2*np.log(2*np.pi)#Step 1 assert float value
        print("return value: " + str(ret))
        return ret[0][0]
        raise NotImplementedError
