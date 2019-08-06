import time
from os import path

import numpy as np

from plot import plot_kde_1d
from run_tests_for_mcmc_on_logistic_fn import run_hmc, run_rwm, run_rmhmc
from utils import get_kdes_of_chain, get_kl_div_for_univariate_pdfs


def run_direct_kl_calculations():
    gt_chain_path = "gt_hmc_10000_w_burnin.npy"  # TODO: Add burn_in or remove from name
    if path.exists(gt_chain_path):
        gt_chain = np.load(gt_chain_path)
        print(gt_chain[0])
    else:
        gt_chain = run_hmc(plot=False, its=10000)
        print(gt_chain[0])
        np.save(gt_chain_path, gt_chain)


def run_kl_calculations_via_kde(plot=False):
    gt_chain_path = "gt_rwm_1000000_w_burnin.npy"  # TODO: Add burn_in or remove from name
    if path.exists(gt_chain_path):
        print("Loading samples...")
        gt_chain = np.load(gt_chain_path)
        print(gt_chain[0])
    else:
        print("Creating samples...")
        gt_chain = run_rwm(plot=False, its=1000000)
        print(gt_chain[0])
        np.save(gt_chain_path, gt_chain)
    gt_kdes = get_kdes_of_chain(gt_chain)

    rwm_chain = run_rwm(its=1000)
    rwm_kdes = get_kdes_of_chain(rwm_chain)

    param_names = ["$\\kappa$", "$\\alpha$", "$\\sigma$"]
    for chain, kdes in [(rwm_chain,rwm_kdes)]:
        for gt_kde, mcmc_kde, i, param_name in zip(gt_kdes, kdes, range(chain.shape[1]),param_names):
            if i==2:
                continue
            trace = chain[:, i]
            max_x= max(trace)
            min_x= min(trace)
            diff_x = max_x-min_x
            print("min",min_x,"max",max_x)
            print(get_kl_div_for_univariate_pdfs(gt_kde, mcmc_kde, min_x=min_x-diff_x/2, max_x=max_x+diff_x/2))

            plot_kde_1d(trace, gt_kde, mcmc_kde, param_name=param_name)
            time.sleep(3)
    exit()
    hmc_chain = run_hmc(its=1000)
    hmc_kdes = get_kdes_of_chain(hmc_chain)

    rmhmc_chain = run_rmhmc(its=1000)
    rmhmc_kdes = get_kdes_of_chain(rmhmc_chain)

if __name__ == '__main__':
    run_kl_calculations_via_kde(plot=True)