import time
import numpy as np
from mcmc import rmhmc, rwm, hmc
from plot import show_mcmc_chain
from utils import get_effective_n
from pints.toy import NealsFunnelLogPDF


np.random.seed(3)

def run_rmhmc_on_neals_funnel(its=10000, plot=False, use_tqdm=True):
    print("Running RMHMC on Neal's Funnel")
    nu_sigma = 3
    nu_mu = 0
    x_mu = 0

    n_frog_its = 6
    eps = 1/6

    init_pt = np.array([3.2 for _ in range(100)]+[0], dtype=np.float64)

    def logl(th):
        ll = 0
        xs, nu = th[:-1], th[-1]
        x_sigma = np.exp(nu / 2)

        ll += -(nu-nu_mu)**2/(2*nu_sigma**2)-1/2*np.log(2*np.pi*nu_sigma**2)
        for x in xs:
            ll += -(x-x_mu)**2/(2*x_sigma**2) - 1/2*np.log(2*np.pi*x_sigma**2)
        return ll

    def dlogldt(th):
        dlldt = np.zeros(shape=th.shape)
        xs, nu = th[:-1], th[-1]
        x_sigma = np.exp(nu / 2)

        dlldt[-1] = -(nu-nu_mu)/(nu_sigma**2)
        for i, x in enumerate(xs):
            dlldt[i] = -(x-x_mu)/(x_sigma**2)
        return dlldt

    def fi_metric(th):
        raise NotImplementedError
        return th

    deriv_fi_metric = lambda *_ : np.array([[[0]]])

    chain, chain_jumps, intermediate_chain = rmhmc(logl, dlogldt, logp_dist=None, init_pt=init_pt, metric=fi_metric,
                                                   deriv_metric=deriv_fi_metric, its=its, n_frog_its=n_frog_its,
                                                   eps=eps, w_noise=False, use_tqdm=use_tqdm)





    avg_chain_jumpprob = np.average(chain_jumps)
    eff_n = get_effective_n(chain)
    num_samples = chain.shape[0]
    print("Avg of jumpprobs", avg_chain_jumpprob)
    print("ESS", eff_n)
    print("Num samples from dist:", num_samples)

    if plot:
        true_fn = lambda x: 1 / (nu_sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - nu_mu) ** 2 / (2 * nu_sigma ** 2))
        plot_title = {}
        plot_title["Avg jumprob"] = round(avg_chain_jumpprob, 5)
        plot_title["ESS"] = np.around(eff_n[-1])
        show_mcmc_chain(chain[:, -1], param_name="$\\nu", true_fn=true_fn, plot_title=plot_title)
    return chain


def run_rwm_on_neals_funnel(plot=False, its=1000):
    print("Running RWM on Neal's Funnel")
    nu_sigma = 3
    nu_mu = 0
    x_mu = 0

    def logl(th):
        ll = 0
        xs, nu = th[:-1], th[-1]
        x_sigma = np.exp(nu / 2)

        ll += -(nu-nu_mu)**2/(2*nu_sigma**2)-1/2*np.log(2*np.pi*nu_sigma**2)
        for x in xs:
            ll += -(x-x_mu)**2/(2*x_sigma**2) - 1/2*np.log(2*np.pi*x_sigma**2)
        return ll

    init_pt = np.array([3.2 for _ in range(100)] + [2.5])

    chain, chain_jumps = rwm(logl, logp_dist=None, init_pt=init_pt, its=its, rw_sigma_ratio=0.01)

    avg_chain_jumpprob = np.average(chain_jumps)
    std_eff_n = get_effective_n(chain, use_standard_ess=True)
    eff_n = get_effective_n(chain)
    num_samples = chain.shape[0]
    print("Avg of jumpprobs", avg_chain_jumpprob)
    print("Standard ESS", std_eff_n)
    print("ESS", eff_n)
    print("Num samples from dist:", num_samples)
    if plot:
        true_fn = lambda x: 1 / (nu_sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - nu_mu) ** 2 / (2 * nu_sigma ** 2))
        plot_title = {}
        plot_title["Avg jumprob"] = round(avg_chain_jumpprob, 5)
        plot_title["Standard ESS"] = np.around(std_eff_n[-1])
        show_mcmc_chain(chain[:, -1], param_name="$\\nu", true_fn=true_fn, plot_title=plot_title)
    return chain


def run_hmc_on_neals_funnel(plot=True, its=1000):
    print("Running HMC on Neal's Funnel")

    nu_sigma = 3
    nu_mu = 0
    x_mu = 0

    n_frog_its = 6
    eps = 1/6
    stepsize_scales = 1.  # As in Betancourt 2015


    def logl(th):
        ll = 0
        xs, nu = th[:-1], th[-1]
        x_sigma = np.exp(nu / 2)

        ll += -(nu-nu_mu)**2/(2*nu_sigma**2)-1/2*np.log(2*np.pi*nu_sigma**2)
        for x in xs:
            ll += -(x-x_mu)**2/(2*x_sigma**2) - 1/2*np.log(2*np.pi*x_sigma**2)
        return ll

    def dlogldt(th):
        dlldt = np.zeros(shape=th.shape)
        xs, nu = th[:-1], th[-1]
        x_sigma = np.exp(nu / 2)

        dlldt[-1] = -(nu-nu_mu)/(nu_sigma**2)
        for i, x in enumerate(xs):
            dlldt[i] = -(x-x_mu)/(x_sigma**2)
        return dlldt

    U = lambda th: - (logl(th))
    dUdT = lambda th: -dlogldt(th)

    init_pt = np.array([3.2 for i in range(100)] + [2.5])

    chain, chain_jumps = hmc(init_pt, U, dUdT, logp_dist=None, its=its,
                             n_frog_its=n_frog_its, eps=eps, stepsize_scales=stepsize_scales)

    avg_chain_jumpprob = np.average(chain_jumps)
    eff_n = get_effective_n(chain)
    num_samples = chain.shape[0]
    print("Avg of jumpprobs", avg_chain_jumpprob)
    print("ESS", eff_n)
    print("Num samples from dist:", num_samples)

    if plot:
        true_fn = lambda x: 1 / (nu_sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - nu_mu) ** 2 / (2 * nu_sigma ** 2))
        plot_title = {}
        plot_title["Avg jumprob"] = round(avg_chain_jumpprob, 5)
        plot_title["ESS"] = np.around(eff_n[-1])
        show_mcmc_chain(chain[:, -1], param_name="$\\nu", true_fn=true_fn, plot_title=plot_title)
    return chain


if __name__ == '__main__':
    #run_rwm_on_neals_funnel(plot=True, its=2500)
    run_hmc_on_neals_funnel(plot=True, its=10000)
    #run_rmhmc_on_neals_funnel()
    exit()
