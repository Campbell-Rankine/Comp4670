import numpy as np

################################################################
##### BLR Question Code
################################################################

def single_EM_iter_blr(features, targets, alpha_i, beta_i):
    # Given the old alpha_i and beta_i, computes expectation of latent variable w: M_n and S_n,
    # and using that computes the new alpha and beta values.
    # Should return M_n, S_n, new_alpha, new_beta in that order, with return shapes (M,1), (M,M), None, None
    ### CODE HERE ###
    fs = features.shape
    ts = targets.shape
    sni = alpha_i*np.eye(fs[1]) + beta_i*(features.T@features)
    sn = np.linalg.inv(sni)
    print(sn)
    print(sn.shape)
    mn = (beta_i*sn)@features.T@targets
    Ed1 = mn.T@mn + np.trace(sni)
    Ed2 = (targets-features @ mn).T @ (targets-features @ mn) + np.trace(features.T @ features @ sn)
    
    new_alpha = (fs[1] / Ed1)[0][0]
    new_beta = (fs[0] / Ed2)[0][0]
    
    print(new_alpha, new_beta)
    return mn, sn, new_alpha, new_beta # (M,1), (M,M), None, None