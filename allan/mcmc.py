import copy

import numpy as np
from numpy.linalg import lstsq
from autograd import jacobian
from tqdm import tqdm
from scipy.optimize import root


def metropolis_step(theta0, proposal_sigma, logl_dist, logp_dist, only_sigma=False):
    if only_sigma:
        proposal_sigma[:-1] = 0
    theta1 = np.random.normal(theta0, proposal_sigma)
    theta0_logl = logl_dist(theta0)
    theta1_logl = logl_dist(theta1)


    theta0_logp = logp_dist(theta0) if logp_dist is not None else 0
    theta1_logp = logp_dist(theta1) if logp_dist is not None else 0

    # Calculate the ratio
    r = np.exp(theta1_logl - theta0_logl + theta1_logp - theta0_logp)

    # Compare r with a uniformly-distributed number between 0 and 1.
    if r >= np.random.uniform():
        theta0 = theta1

    return theta0, r


def rwm(logl_dist, logp_dist, init_pt, its=1000, burn_in_ratio=0.0, rw_sigma_ratio=0.01):
    chain = np.empty((its, len(init_pt)), dtype=float)
    chain_jumps = np.empty(its, dtype=float)
    theta = init_pt
    if np.any(init_pt == 0):
        proposal_sigma = np.abs(init_pt*rw_sigma_ratio)+.1
    else:
        proposal_sigma = np.abs(init_pt * rw_sigma_ratio)
    for t in tqdm(range(its), disable=False):
        # Propose a new location from a jumping distribution
        theta, r = metropolis_step(theta, proposal_sigma, logl_dist, logp_dist)
        chain_jumps[t] = min(r, 1)
        chain[t] = theta

    return chain[int(burn_in_ratio*its):, ], chain_jumps[int(burn_in_ratio*its):]


def hmc_proposal(theta0, momentum0, n_iterations, eps, dUdT, logp_dist):
    """
    Leapfrog integration to propose new theta (position) and momentum
    Note;
        Keeps H roughly constant even under numerical error
    """

    # Init theta and momentum
    theta = copy.copy(theta0)
    momentum = copy.copy(momentum0)

    # First momentum update
    momentum -= eps/2.0*dUdT(theta)

    for i in range(n_iterations - 1):
        dtheta = eps*momentum
        theta += dtheta

        dmomentum = -eps * dUdT(theta)
        momentum += dmomentum

    # Final leapfrog updates
    theta += eps * momentum
    momentum -= eps/2.0 * dUdT(theta)

    momentum = -momentum  # To make proposal symmetric -- Not actually necessary
    if logp_dist is not None and logp_dist(theta) < -1e100:
        return theta0, momentum0  # Reject theta and momentum
    return theta, momentum


def hmc(init_pt, U, dUdT, logp_dist, stepsize_scales=None, its=1000,
        burn_in_ratio=0.0, n_frog_its=6, eps=0.5):

    stepsize_scales = stepsize_scales if stepsize_scales is not None else np.abs(init_pt/100)

    chain = np.empty((its, len(init_pt)), dtype=float)
    chain_jumps = np.empty(its, dtype=float)
    theta0 = init_pt

    K = lambda mom: np.dot(mom, mom) / 2.0

    for t in tqdm(range(its), disable=False):
        # Propose a new location from a jumping distribution
        momentum0 = np.random.normal(size=theta0.size)
        scaled_eps = eps*stepsize_scales
        theta1, momentum1 = hmc_proposal(theta0, momentum0, n_frog_its, scaled_eps, dUdT, logp_dist)

        # Compute (log) likelihoods and priors
        U_0 = U(theta0)
        K_0 = K(momentum0)
        U_1 = U(theta1)
        K_1 = K(momentum1)


        # Calculate the ratio
        r = np.exp(U_0+K_0 -U_1-K_1)
        #print("r")
        #print(r)

        # Compare r with a uniformly-distributed number between 0 and 1.
        jump = 0
        if r >= np.random.uniform():
            theta0 = theta1
            jump = 1
            # Add to chain of samples
        chain_jumps[t] = min(r, 1)
        chain[t] = theta0

    return chain[int(burn_in_ratio*its):, ], chain_jumps[int(burn_in_ratio)*its:]


def get_momentum_root(theta, mom_old, sigma, eps, dHdT):
    fun = lambda mom: mom_old - mom - eps / 2 * dHdT(theta, mom, sigma)
    new_m = root(x0=mom_old, fun=fun, tol=1e-5, method="lm")  # options={"maxiter": 100000}
    # print(new_m.nfev)
    if not new_m.success:
        print("dHdT(theta, mom_old, sigma)", dHdT(theta, mom_old, sigma))
        print( "- eps / 2 * dHdT(theta, mom_old, sigma)", - eps / 2 * dHdT(theta, mom_old, sigma))
        print("theta, mom_old, sigma, eps, dHdT", theta, mom_old, sigma, eps, dHdT)
        raise ValueError("Momentum root not converging - " + new_m.message)
    return new_m.x, new_m.nfev


def get_theta_root(theta_old, momentum, sigma, eps, dHdp):
    fun = lambda th: theta_old - th + eps / 2 * (dHdp(theta_old, momentum, sigma) + dHdp(th, momentum, sigma))
    new_th = root(x0=theta_old, fun=fun, tol=1e-5, method="lm")  # options={"maxiter": 100000}
    if not new_th.success:
        raise ValueError("Theta root not converging" + new_th.message)
    return new_th.x, new_th.nfev


def rmhmc_proposal(theta0, momentum0, sigma, dHdT, dHdp, logp_dist, n_frog_iterations, eps):
    theta1, momentum1 = copy.copy(theta0), copy.copy(momentum0)
    intermediate_vals = np.empty(shape=(n_frog_iterations+1, 2, theta1.size))
    avg_th_nfev = 0
    avg_mom_nfev = 0

    intermediate_vals[0, 0] = theta1
    intermediate_vals[0, 1] = momentum1

    for i in range(n_frog_iterations):
        momentum1, mom_nfev = get_momentum_root(theta1, momentum1, sigma, eps, dHdT)  # Eq. 16 Girolami2011
        theta1, th_nfev = get_theta_root(theta1, momentum1, sigma, eps, dHdp)  # Eq 17, Girolami2011
        momentum1 -= eps/2*dHdT(theta1, momentum1, sigma)  # Eq 18. Girolami2011

        avg_th_nfev += th_nfev
        avg_mom_nfev += mom_nfev

        intermediate_vals[i+1, 0] = theta1
        intermediate_vals[i+1, 1] = momentum1


    avg_th_nfev /= n_frog_iterations
    avg_mom_nfev /= n_frog_iterations

    momentum1 = -momentum1  # To make proposal symmetric TODO: Still true for rmhmc?

    if logp_dist is not None and logp_dist(theta1) < -1e100:
        return theta0, momentum0, avg_th_nfev, avg_mom_nfev, intermediate_vals
    return theta1, momentum1, avg_th_nfev, avg_mom_nfev, intermediate_vals


def rmhmc(logl, dlogldt, logp_dist, init_pt, metric, deriv_metric, its=100, burn_in_ratio=0.0, n_frog_its=10,
          eps=.5, w_noise=True, exact_proposal=None, use_tqdm=True):
    """
    :param w_noise: if w_noise, then last param is sigma. will be estimated w/ RWM
    :param exact_proposal:
    :param use_tqdm: If no tqdm and exact prop, will compare with leapfrog for each it
    :return:
    """

    chain = np.empty((its, len(init_pt)), dtype=float)
    num_rmhmc_params = init_pt.size-1 if w_noise else init_pt.size
    intermediate_chain = np.empty((its, n_frog_its + 1, 2, num_rmhmc_params))
    chain_jumps = np.empty(its, dtype=float)
    theta0 = init_pt
    rwm_proposal_sigma = init_pt/10

    def H(th, p, sigma):
        h = 0
        G = metric(th, sigma)
        inv_G = np.linalg.inv(G)

        th_w_sigma = np.append(th, sigma)

        h += -logl(th_w_sigma)

        h += 1/2*np.log(np.linalg.det(G)*(2*np.pi)**len(th))
        h += 1/2*np.inner(p, np.matmul(inv_G, p))
        return h

    def dHdT(th, p, sigma):
        dH = np.zeros(shape=th.size)
        th_w_sigma = np.append(th, sigma)
        dH += -dlogldt(th_w_sigma)[:-1]  # todo: fix hack

        inv_G = np.linalg.inv(metric(th, sigma))
        deriv_G = deriv_metric(th, sigma)
        for i in range(th.size):

            deriv_G_i = deriv_G[:, :, i]

            inv_G_deriv_G = np.matmul(inv_G, deriv_G_i)
            inv_G_deriv_G_inv_G = np.matmul(inv_G_deriv_G, inv_G)
            p_inv_G_deriv_G_inv_G = np.matmul(p, inv_G_deriv_G_inv_G)

            diff1 = 1/2*np.trace(inv_G_deriv_G)
            diff2 = -1/2*np.matmul(p_inv_G_deriv_G_inv_G, p)

            dH[i] += diff1
            dH[i] += diff2
        return dH

    def dHdp(th, p, sigma):
        g = metric(th, sigma)
        x, _, _, _ = lstsq(g, p, rcond=None)
        return x

    for t in tqdm(range(its), disable=not use_tqdm):

        ### PT 1 - RMHMC for model params ###
        if not use_tqdm:
            print("Iteration", t)
        if w_noise:
            theta0, sigma = theta0[:-1], theta0[-1]
        else:
            sigma = float("nan")
        #G_theta = np.linalg.inv(metric(theta0, sigma))  # TODO!
        G_theta = metric(theta0, sigma)  # TODO!
        momentum0 = np.random.multivariate_normal(mean=np.zeros(shape=theta0.shape),
                                                  cov=G_theta)

        if exact_proposal is None:
            theta1, momentum1, _,_, intermediate_vals = rmhmc_proposal(theta0, momentum0, sigma, dHdT, dHdp, logp_dist,
                                                                       n_frog_its, eps)
        else:
            theta1, momentum1 = exact_proposal(theta0, momentum0, t=n_frog_its * eps)

            if not use_tqdm:
                theta1_num, momentum1_num, _, _, intermediate_vals = rmhmc_proposal(theta0, momentum0, sigma, dHdT,
                                                                                    dHdp, logp_dist,
                                                                                    n_frog_its, eps)
                rel_th = np.abs((theta1_num - theta1) / theta1)
                rel_mom = np.abs((momentum1_num - momentum1) / momentum1)
                if False and (np.any(rel_th>0.1) or np.any(rel_mom>0.1)):  # TODO
                    print()
                    print("th - diff:", theta1_num - theta1, "exact:", theta1, "rel:", rel_th)
                    print("mom - diff:", momentum1_num - momentum1, "exact", momentum1, "rel:", rel_mom)

                    print("th-steps")
                    print(intermediate_vals[:,0,:])
                    print("mom-steps")
                    print(intermediate_vals[:,1,:])

        # TODO analyse nfev

        H_0 = H(theta0, momentum0, sigma)
        H_1 = H(theta1, momentum1, sigma)

        # Calculate the ratio

        r = np.exp(H_0-H_1)
        #print(r)

        # Compare r with a uniformly-distributed number between 0 and 1.

        jump_rmhmc = 0
        if r >= np.random.uniform():
            theta0 = theta1
            if not (use_tqdm and exact_proposal is not None):
                intermediate_chain[t,:,:,:] = intermediate_vals

            jump_rmhmc = 1
            # Add to chain of samples
        if w_noise:
            theta0 = np.append(theta0, sigma)
            # RWM-step for noise param
            theta0, _ = metropolis_step(theta0, rwm_proposal_sigma, logl, logp_dist, only_sigma=True)  # TODO: Reintroduce

        chain_jumps[t] = min(r, 1)
        chain[t] = theta0
        if not use_tqdm:
            print("theta0", theta0)
    return chain[int(burn_in_ratio*its):, ], chain_jumps[int(burn_in_ratio)*its:], intermediate_chain[int(burn_in_ratio)*its:,:,:,:]
