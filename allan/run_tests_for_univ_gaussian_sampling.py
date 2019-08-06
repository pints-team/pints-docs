import numpy as np

from mcmc import rmhmc, rwm, hmc
from plot import show_mcmc_chain, plot_path_in_pos_mom_space
from utils import get_effective_n


def run_rwm_for_univ_gaussian_sampling(plot=False, its=1000):
    print("Running RWM on Univ Gauss Sampling")
    gt_params = {"mu": -2, "sigma": np.sqrt(0.5)}
    mu = gt_params["mu"]
    sigma = gt_params["sigma"]

    logl = lambda th: -(th[0] - mu) ** 2 / (2 * sigma ** 2) - 1 / 2 * np.log(2 * np.pi * sigma ** 2)
    init_pt = np.array([mu])

    chain, chain_jumps = rwm(logl, logp_dist=None, init_pt=init_pt, its=its, rw_sigma_ratio=sigma)

    avg_chain_jumpprob = np.average(chain_jumps)
    std_eff_n = get_effective_n(chain, use_standard_ess=True)
    eff_n = get_effective_n(chain)
    num_samples = chain.shape[0]
    print("Avg of jumpprobs", avg_chain_jumpprob)
    print("Standard ESS", std_eff_n)
    print("ESS", eff_n)
    print("Num samples from dist:", num_samples)
    if plot:
        true_fn = lambda x: 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))
        plot_title = {}
        plot_title["Avg jumprob"] = round(avg_chain_jumpprob, 5)
        plot_title["Standard ESS"] = np.around(std_eff_n)
        show_mcmc_chain(chain[:, 0], param_name="$\\theta$", true_fn=true_fn, plot_title=plot_title)
    return chain


def run_hmc_for_univ_gaussian_sampling(plot=True, its=10):
    print("Running HMC on Univ Gauss Sampling")
    gt_params = {"mu": 2., "sigma": np.sqrt(0.5)}
    mu = gt_params["mu"]
    sigma = gt_params["sigma"]
    n_frog_its = 6
    eps = .5
    stepsize_scales = .1

    logl = lambda th: -(th[0] - mu) ** 2 / (2 * sigma ** 2) - 1 / 2 * np.log(2 * np.pi * sigma ** 2)
    dlogldt = lambda th: np.array([-(th[0] - mu) / sigma ** 2])

    init_pt = np.array([mu])

    U = lambda th: - (logl(th))
    dUdT = lambda th: -dlogldt(th)

    chain, chain_jumps = hmc(init_pt, U, dUdT, logp_dist=None, its=its,
                             n_frog_its=n_frog_its, eps=eps, stepsize_scales=stepsize_scales)

    avg_chain_jumpprob = np.average(chain_jumps)
    eff_n = get_effective_n(chain)
    num_samples = chain.shape[0]
    print("Avg of jumpprobs", avg_chain_jumpprob)
    print("ESS", eff_n)
    print("Num samples from dist:", num_samples)
    if plot:
        true_fn = lambda x: 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))
        plot_title = {}
        plot_title["Avg jumprob"] = round(avg_chain_jumpprob, 5)
        plot_title["ESS"] = np.around(eff_n)
        show_mcmc_chain(chain[:, 0], param_name="$\\theta$", true_fn=true_fn, plot_title=plot_title)
    return chain


def run_exact_rmhmc_for_univ_gaussian_sampling(plot=False, its=1000, use_tqdm=False):
    print("Running Exact RMHMC on Univ Gauss Sampling")
    gt_params = {"mu": -2, "sigma": np.sqrt(.2)}
    mu = gt_params["mu"]
    sigma = gt_params["sigma"]
    n_frog_its = 4
    eps = .2

    logl = lambda th: -(th[0] - mu) ** 2 / (2 * sigma ** 2) - 1 / 2 * np.log(2 * np.pi * sigma ** 2)
    dlogldt = lambda th: np.array([-1 / sigma ** 2 * (th[0] - mu), 0])  # TODO: fix use of 0 as a hack

    fi_metric = lambda *_ : np.array([[1/sigma**2]])

    deriv_fi_metric = lambda *_ : np.array([[[0]]])

    def exact_proposal(th, mom, t):
        c1 = mom
        c2 = th - mu
        th_new = c1 * sigma ** 2 * np.sin(t) + c2 * np.cos(t) + mu
        mom_new = c1 * np.cos(t) - c2 * np.sin(t) / (sigma ** 2)
        mom_new = -mom_new
        return th_new, mom_new

    init_pt = np.array([mu])

    chain, chain_jumps, intermediate_chain = rmhmc(logl, dlogldt, logp_dist=None, init_pt=init_pt, metric=fi_metric,
                                                   deriv_metric=deriv_fi_metric, its=its, n_frog_its=n_frog_its,
                                                   eps=eps, w_noise=False, exact_proposal=exact_proposal,
                                                   use_tqdm=use_tqdm)

    print("Avg of jumpprobs", np.average(chain_jumps))
    print("ESS:", get_effective_n(chain))
    print("Num samples from dist:", chain.shape[0])
    if plot:
        true_fn= lambda x: 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))
        trace = chain[:, 0]
        num_trace = intermediate_chain[:, -1, 0, :]
        show_mcmc_chain(trace, param_name="$\\theta$", true_fn=true_fn)
        if not use_tqdm:
            show_mcmc_chain(num_trace, param_name="$\\theta\_num$", true_fn=true_fn)
            plot_path_in_pos_mom_space(intermediate_chain, exact_proposal, t=n_frog_its*eps)
    return chain


def run_rmhmc_for_univ_gaussian_sampling(plot=False, its=1000,use_tqdm=False):
    print("Running RMHMC on Univ Gauss Sampling")
    gt_params = {"mu": -2, "sigma": np.sqrt(0.2)}
    mu = gt_params["mu"]
    sigma = gt_params["sigma"]
    n_frog_its = 4
    eps = .2

    logl = lambda th: -(th[0] - mu) ** 2 / (2 * sigma ** 2) - 1 / 2 * np.log(2 * np.pi * sigma ** 2)
    dlogldt = lambda th: np.array([-1 / sigma ** 2 * (th[0] - mu), 0])  # TODO: fix use of 0 as a hack

    fi_metric = lambda *_ : np.array([[1/sigma**2]])

    deriv_fi_metric = lambda *_ : np.array([[[0]]])

    init_pt = np.array([mu], dtype=np.float64)

    chain, chain_jumps, intermediate_chain = rmhmc(logl, dlogldt, logp_dist=None, init_pt=init_pt, metric=fi_metric,
                                                   deriv_metric=deriv_fi_metric, its=its, n_frog_its=n_frog_its, eps=eps, w_noise=False, use_tqdm=use_tqdm)

    avg_chain_jumpprob = np.average(chain_jumps)
    eff_n = get_effective_n(chain)
    num_samples = chain.shape[0]
    print("Avg of jumpprobs", avg_chain_jumpprob)
    print("ESS", eff_n)
    print("Num samples from dist:", num_samples)
    if plot:
        true_fn = lambda x: 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))
        plot_title = {}
        plot_title["Avg jumprob"] = round(avg_chain_jumpprob, 5)
        plot_title["ESS"] = np.around(eff_n)
        show_mcmc_chain(chain[:, 0], param_name="$\\theta$", true_fn=true_fn, plot_title=plot_title)
    return chain




if __name__ == '__main__':
    #chain = run_rmhmc_for_univ_gaussian_sampling(plot=True, its=10000, use_tqdm=True)
    run_hmc_for_univ_gaussian_sampling(plot=True, its=10000)
    #run_rwm_for_univ_gaussian_sampling(plot=True, its=100000)
    # chain = run_exact_rmhmc_for_univ_gaussian_sampling(plot=True, its=25000, use_tqdm=True)
