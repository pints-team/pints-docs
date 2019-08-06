import time

import numpy as np
from examples.divcode.mcmc import rwm, hmc, rmhmc
from examples.divcode.plot import show_mcmc_chain, plot_data, show_trace
from examples.divcode.utils import get_noisy_logistic_samples, noisy_logistic_log_likelihood, get_log_unif_prior, \
    dnoisy_logistic_log_likelihood_dtheta, get_dUdT, get_effective_n, \
    get_deriv_of_fim_w_gauss_assump, get_fim_w_gauss_assump, logistic_fn, dlogisticdt, d2logisticdt2, get_dlogldt, \
    get_logl

np.random.seed(1)


def run_rwm(plot=False, its=1000):
    print("Running RWM on Logistic Function")
    gt_params = {"kappa": 500, "alpha": 0.015, "sigma": 20}
    unif_prior_params = [(0, 1000), (0, 1000), (0, 1000)]
    num_samples = 100
    rw_sigma_ratio = 0.01

    xs, ys = get_noisy_logistic_samples(num_samples=num_samples, **gt_params)
    data = np.transpose(np.vstack((xs, ys)))

    if plot:
        true_fn = lambda x: logistic_fn(x, kappa=gt_params["kappa"], alpha=gt_params["alpha"])
        plot_data(xs, ys, true_fn=true_fn)
        #true_fn_2 = lambda x: logistic_fn(x, kappa=502, alpha=0.01505)
        #plot_data(xs,ys,true_fn=true_fn_2)

    init_pt = np.array([gt_params["kappa"]*1., gt_params["alpha"]*1., gt_params["sigma"]*1.])

    logp_dist = get_log_unif_prior(*unif_prior_params)
    logl = get_logl(noisy_logistic_log_likelihood, data)
    chain, chain_jumps = rwm(logl, logp_dist, init_pt=init_pt, its=its, rw_sigma_ratio=rw_sigma_ratio)

    print("Ratio of accepted proposed states", np.average(chain_jumps))
    print("ESS:", get_effective_n(chain))
    print("Num samples from dist:", chain.shape[0])

    if plot:
        for i, name in zip(range(len(init_pt)), ["$\\kappa$", "$\\alpha$", "$\\sigma$"]):
            show_mcmc_chain(chain[:, i], param_name=name)
        show_trace(chain)
    return chain


def run_hmc(plot=False, its=1000):
    print("Running HMC on Logistic Function")
    gt_params = {"kappa": 500, "alpha": 0.015, "sigma": 20}
    unif_prior_params = [(0, 1000), (0, 1000), (0, 1000)]
    num_samples = 100
    n_frog_its = 6
    eps = 1/6

    xs, ys = get_noisy_logistic_samples(num_samples=num_samples, **gt_params)
    data = np.transpose(np.vstack((xs, ys)))

    if plot:
        true_fn = lambda x: logistic_fn(x, kappa=gt_params["kappa"], alpha=gt_params["alpha"])
        plot_data(xs, ys, true_fn=true_fn)

    dUdT = get_dUdT(dnoisy_logistic_log_likelihood_dtheta, data)  # TODO/NOTE: should technically have prior too, but deriv==0

    init_pt = np.array([gt_params["kappa"]*1, gt_params["alpha"]*1, gt_params["sigma"]*1])
    logp_dist = get_log_unif_prior(*unif_prior_params)
    lldist = noisy_logistic_log_likelihood

    U = lambda theta: -sum([lldist(sample, *theta) for sample in data])-logp_dist(theta)


    #logp_dist, init_pt, U, dUdT,
    chain, chain_jumps = hmc(init_pt, U, dUdT, logp_dist, its=its, n_frog_its=n_frog_its, eps=eps)

    print("Ratio of accepted proposed states", np.average(chain_jumps))
    print("ESS:", get_effective_n(chain))
    print("Num samples from dist:", chain.shape[0])

    if plot:
        for i, name in zip(range(len(init_pt)), ["$\\kappa$", """$\\alpha$""", "$\\sigma$"]):
            show_mcmc_chain(chain[:, i], param_name=name)
        show_trace(chain)
    return chain


def run_rmhmc(plot=False, its=1000, use_tqdm=True):
    print("Running RMHMC on Logistic Function")
    gt_params = {"kappa": 500, "alpha": 0.015, "sigma": 20}
    unif_prior_params = [(0, 1000), (0, 1000), (0, 1000)]
    num_samples = 100
    n_frog_its = 6
    eps = 1/6

    xs, ys = get_noisy_logistic_samples(num_samples=num_samples, **gt_params)
    data = np.transpose(np.vstack((xs, ys)))

    if plot:
        true_fn = lambda x: logistic_fn(x, kappa=gt_params["kappa"], alpha=gt_params["alpha"])
        plot_data(xs, ys, true_fn=true_fn)

    dlogldt = get_dlogldt(dnoisy_logistic_log_likelihood_dtheta, data)  # TODO/NOTE: should technically have prior too, but deriv==0
    logl = get_logl(noisy_logistic_log_likelihood, data)

    fi_metric = get_fim_w_gauss_assump(dlogisticdt, data)
    deriv_fi_metric = get_deriv_of_fim_w_gauss_assump(dlogisticdt, d2logisticdt2, data)

    init_pt = np.array([gt_params["kappa"], gt_params["alpha"], gt_params["sigma"]])
    logprior = get_log_unif_prior(*unif_prior_params)

    chain, chain_jumps, intermediate_chain = rmhmc(logl, dlogldt, logp_dist=logprior, init_pt=init_pt, metric=fi_metric,
                                                   deriv_metric=deriv_fi_metric, its=its, n_frog_its=n_frog_its, eps=eps,
                                                   use_tqdm=use_tqdm, w_noise=True)
    time.sleep(0.1)
    print("Avg of jumpprobs", np.average(chain_jumps))
    print("ESS:", get_effective_n(chain))
    print("Num samples from dist:", chain.shape[0])
    if plot:
        for i, name in zip(range(len(init_pt)), ["$\\kappa$", "$\\alpha$", "$\\sigma$"]):
            show_mcmc_chain(chain[:, i], param_name=name)
        show_trace(chain)
    return chain


if __name__ == '__main__':
    #run_rwm(plot=True, its=10000)
    #run_hmc(plot=True, its=10000)

    run_rmhmc(plot=True, its=100, use_tqdm=True)
    pass
