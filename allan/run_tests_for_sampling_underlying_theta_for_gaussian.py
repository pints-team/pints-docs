import time

import numpy as np

from mcmc import rmhmc, hmc, rwm
from plot import show_mcmc_chain
from utils import get_effective_n, get_log_unif_prior, get_normal_samples, get_dUdT

# np.random.seed(2)

def run_rwm_for_sampling_underlying_theta_for_gaussian(plot=False, its=10000):
    print("Running RWM on Gaussian with underlying theta")

    gt_params = {"theta": np.sqrt(2), "sigma": 2}
    mu_fn = lambda th: th ** 2
    unif_prior_params = [(-1000, 1000)]
    gt_theta = gt_params["theta"]
    gt_sigma = gt_params["sigma"]
    num_samples = 100
    rw_sigma_ratio = 0.1

    samples = get_normal_samples(num_samples=num_samples, mu=mu_fn(gt_theta), sigma=gt_sigma)

    logl = lambda th: sum([-(x - mu_fn(th[0])) ** 2 / (2 * gt_sigma ** 2) - 1 / 2 * np.log(2 * np.pi * gt_sigma ** 2)
                           for x in samples])
    logp_dist = get_log_unif_prior(*unif_prior_params)
    init_pt = np.array([gt_theta])

    chain, chain_jumps = rwm(logl, logp_dist, init_pt=init_pt, its=its, rw_sigma_ratio=rw_sigma_ratio)

    time.sleep(0.1)
    print("Avg of jumpprobs", np.average(chain_jumps))
    print("ESS:", get_effective_n(chain))
    print("Num samples from dist:", chain.shape[0])
    if plot:
        show_mcmc_chain(chain[:, 0], param_name="$\\theta$")
    return chain



def run_hmc_for_sampling_underlying_theta_for_gaussian(plot=False, its=1000):
    print("Running HMC on Gaussian with underlying theta")

    gt_params = {"theta": np.sqrt(5), "sigma": 2}
    mu_fn = lambda th: th ** 2
    unif_prior_params = [(-1000, 1000)]
    n_frog_its = 4
    eps = .2
    gt_theta = gt_params["theta"]
    gt_sigma = gt_params["sigma"]
    num_samples = 100

    samples = get_normal_samples(num_samples=num_samples, mu=mu_fn(gt_theta), sigma=gt_sigma)


    dlogldt = lambda th: np.array([sum([4*th[0]*x-4*th[0]**3 for x in samples])])

    logl = lambda th: sum([-(x-mu_fn(th[0]))**2/(2*gt_sigma**2)-1/2*np.log(2*np.pi*gt_sigma**2) for x in samples])

    logp_dist = get_log_unif_prior(*unif_prior_params)

    U = lambda th: - logl(th) - logp_dist(th)

    dUdT = lambda th: -dlogldt(th)


    init_pt = np.array([gt_theta])

    # logp_dist, init_pt, U, dUdT,
    chain, chain_jumps = hmc(init_pt, U, dUdT, logp_dist, its=its, n_frog_its=n_frog_its, eps=eps)

    time.sleep(0.1)
    print("Avg of jumpprobs", np.average(chain_jumps))
    print("ESS:", get_effective_n(chain))
    print("Num samples from dist:", chain.shape[0])
    if plot:
        show_mcmc_chain(chain[:, 0], param_name="$\\theta$")
    return chain


def run_rmhmc_for_sampling_underlying_theta_for_gaussian(plot=False, its=1000):
    print("Running RMHMC on Gaussian with underlying theta")
    gt_params = {"theta": np.sqrt(5), "sigma": 2}
    mu_fn = lambda th: th**2
    unif_prior_params = [(-1000, 1000)]
    n_frog_its = 6
    eps = .5
    gt_theta = gt_params["theta"]
    gt_sigma = gt_params["sigma"]
    num_samples = 100

    samples = get_normal_samples(num_samples=num_samples, mu=mu_fn(gt_theta), sigma=gt_sigma)

    logp_dist = get_log_unif_prior(*unif_prior_params)

    dlogldt = lambda th: np.array([sum([4*th[0]*x-4*th[0]**3 for x in samples]), 0])  # TODO: fix use of 0 as a hack

    logl = lambda th: sum([-(x-mu_fn(th[0]))**2/(2*gt_sigma**2)-1/2*np.log(2*np.pi*gt_sigma**2) for x in samples])

    fi_metric = lambda th, _ : num_samples*np.array([[4*th[0]**2/gt_sigma**2]])

    deriv_fi_metric = lambda th,_ : num_samples*np.array([[[8*th[0]/gt_sigma**2]]])

    init_pt = np.array([gt_theta])


    chain, chain_jumps, intermediate_chain = rmhmc(logl, dlogldt, logp_dist=logp_dist, init_pt=init_pt, metric=fi_metric,
                               deriv_metric=deriv_fi_metric, its=its, n_frog_its=n_frog_its, eps=eps, w_noise=False)

    time.sleep(0.1)
    print("Avg of jumpprobs", np.average(chain_jumps))
    print("ESS:", get_effective_n(chain))
    print("Num samples from dist:", chain.shape[0])
    if plot:
        show_mcmc_chain(chain[:, 0], param_name="$\\theta$")
    return chain


if __name__ == '__main__':
    #run_rmhmc_for_sampling_underlying_theta_for_gaussian(plot=True, its=10000)
    #run_hmc_for_sampling_underlying_theta_for_gaussian(plot=True, its=10000)
    run_rwm_for_sampling_underlying_theta_for_gaussian(plot=True, its=100000)
