import numpy as np
from functools import lru_cache
from scipy.integrate import quad

################################################################
##### Helper functions (DO NOT CHANGE)
################################################################

@lru_cache  # Makes things go fast
def normalise_expontial_family(sufstat, eta):
    unnorm_prob = lambda z: np.exp(sufstat(z) @ np.array(eta))
    Z, err = quad(unnorm_prob, -np.inf, np.inf)
    return float(Z)

def exponential_family_pdf(x, sufstat, eta):
    # Input Shapes: (1,), None, (M,)
    # sufstat designates the sufficient statistic map for the exponential
    # family, taking values in (1,) to (M,).
    unnorm_prob = lambda z: np.exp(sufstat(z) @ eta)
    eta = eta.squeeze()
    Z = normalise_expontial_family(sufstat, tuple(eta))
    prob = unnorm_prob(x) / Z # Here Z = exp(-psi(eta))
    
    return prob

################################################################
##### EMM Question Code
################################################################

def weighted_probs(data, pi, eta, sufstat, N, K):
    # Input Shapes: (N,), (K,1), (K,m), None, None
    # Should implement pi_k * q(x_n|eta_k) for each n, k, and thus return shape
    # should be (N,K). You should use exponential_family_pdf as defined above.
    # Note: sufstat(x) = u(x).
    # Works for scalars ((1,) -> (2,)); and 1D arrays ((N,) -> (N, 2)).
    ### CODE HERE ###
    probs = np.zeros((N,K))
    for k in range(K):
            probs[:, k] = pi[k] * exponential_family_pdf(data, sufstat, eta[k])
    assert probs.shape == (N,K)
    return probs # (N, K)

def e_step_EMM(data, pi, eta, sufstat, N, K):
    # Input Shapes: (N,), (K,1), (K,m), None, None
    # Should implement gamma_nk for each n, k; and thus return shape should be (N,K).
    # Note: sufstat(x) = u(x).
    # This works for scalars ((1,) -> (2,)); and 1D arrays ((N,) -> (N, 2)).
    # It should use weighted_probs.
    ### CODE HERE ###
    gamma = np.zeros((N,K))
    ### - Get div value - ###
    arr = weighted_probs(data, pi, eta, sufstat, N, K)
    div = np.sum(arr, axis = 1, keepdims = True)
    gamma = arr / div
    assert gamma.shape == (N,K)
    return gamma # (N, K)

def m_step_EMM(data, gamma, sufstat, exp_to_nat, N, K):
    # Input Shapes: (N,D), (N,K), None, None, None
    # Should implement updates for pi, Eta, and return them in that order.
    # exp_to_nat is a function which converts the expectation parameter to
    # natural parameter. This only works dimensions (2,) -> (2,).
    # Note: sufstat(x) = u(x).
    # This works for scalars (1,) -> (2,); and 1D arrays (N,) -> (N, 2).
    # Return shapes should be (K,1), (K,m).
    ### CODE HERE ###
    ###- Calculate pi_new - ###
    pi_new = np.zeros(K)
    assert pi_new.shape == (K,)
    eta_new = []
    for k in range(K):
        pi_new[k] = np.sum(gamma[:, k]) / N
        eta_new.append(exp_to_nat(np.sum(sufstat(data) * gamma[:,k].reshape(N,1), axis = 0) / np.sum(gamma[:,k])))
    eta_new = np.array(eta_new)
    print(eta_new.shape, K)
    return pi_new, eta_new # (K,1), (K,m)