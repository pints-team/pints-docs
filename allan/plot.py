from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation

from scipy.stats import gaussian_kde
import numpy as np


def show_mcmc_chain(chain, param_name="$\theta$", true_fn=None, plot_title=None):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8/1.618))
    if plot_title is not None:
        s = "".join([k+ ":"+ str(v)+" - " for k,v in plot_title.items()])
        s = s[:-2]
        f.suptitle(s)
    ax1.plot(chain)
    ax1.set_xlabel(r'iteration $t$')
    ax1.set_ylabel(r'parameter {}'.format(param_name))
    count, bins, ignored = ax2.hist(chain, bins=50, density=True)
    if true_fn is not None:
        ax2.plot(bins, true_fn(bins), linewidth=2, color='r')
    ax2.set_ylabel(r'probability')
    ax2.set_xlabel(r'parameter {}'.format(param_name))
    ax2.set_xlim(min(chain), max(chain))
    ax1.set_ylim(min(chain), max(chain))
    f.tight_layout(rect=[0, 0.03, 1, 0.9])  # lbrt
    plt.show()


def plot_data(xs, ys, true_fn=None):
    plt.figure()
    if true_fn is not None:
        plt.plot(xs, true_fn(xs))
    plt.plot(xs, ys, "x")
    plt.show()
    plt.close()


def show_trace(chain, chain_jumps=None):
    if chain_jumps is None:
        chain_jumps = np.ones(shape=chain.shape[0])
        #chain_jumps = np.arange(start=0, stop=chain.shape[0], step=1)
    axes = plt.gca()
    axes.set_xlim([min(chain[:,0]), max(chain[:,0])])
    axes.set_ylim([min(chain[:,1]), max(chain[:,1])])
    i = 0.01
    for ((x1,y1,_),(x2,y2,_),r) in zip(chain[:-1], chain[1:], chain_jumps[1:]):
        # i+=1/len(chain)
        plt.plot([x1,x2], [y1,y2],str(i))
    plt.show()
    plt.close()


def plot_path_in_pos_mom_space(intermediate_chain, exact_proposal,t):
    ## (its, n_frog_its + 1, 2, num_rmhmc_params)
    ncols = 2
    nrows = 2

    fig, axs = plt.subplots(nrows=nrows,ncols=ncols)

    for i in range(nrows):
        for j in range(ncols):
            plot_no = i*ncols+j
            ax = axs[(i,j)]
            poss, moms = intermediate_chain[plot_no,:,0,:], intermediate_chain[plot_no,:,1,:]
            start_pos, start_mom = poss[0], moms[0]

            exact_poss, exact_moms = zip(*[exact_proposal(start_pos,start_mom, it/100*t) for it in range(100)])
            exact_moms = [-mom for mom in exact_moms]  # TODO: Fix hack

            ax.plot(poss, moms, color="r")
            ax.plot(exact_poss, exact_moms, color="g")

            ax.plot([exact_poss[0]], [exact_moms[0]], 'bo')
            ax.plot([exact_poss[-1]], [exact_moms[-1]], 'gx')
            ax.plot([poss[-1]], [moms[-1]], 'rx')

            ax.set_xlabel("$\\theta$")
            ax.set_ylabel("$p$")
    plt.show()


def plot_kde_1d(chain, gt_kde, mcmc_kde, param_name="",ax=None):
    """
    Creates a 1d histogram and an estimate of the PDF using KDE.
    Inspired by plot-mcmc-pairwise-kde-plots.ipynb
    """

    if ax is None:
        fig, ax = plt.subplots()
    if param_name:
        ax.set_xlabel("Parameter {}".format(param_name))

    ax.set_ylabel('Probability density')
    xmin = np.min(chain)
    xmax = np.max(chain)
    x1 = np.linspace(xmin, xmax, 100)
    x2 = np.linspace(xmin, xmax, 50)
    mcmc_fn = mcmc_kde(x1)
    gt_fn = gt_kde(x1)
    ax.hist(chain, bins=x2, density=True)
    ax.plot(x1, mcmc_fn, color="r")
    ax.plot(x1, gt_fn, color="g")
    plt.show()


def plot_autocorrelation(trace):
    plt.figure()
    plt.acorr(trace- np.mean(trace))
    plt.xlim(-0.5, len(trace)+0.5)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.show()


def animate_trace(chain, intermediate_steps):
    # First set up the figure, the axis, and the plot element we want to animate
    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = plt.plot([], [], 'ro')

    def init():
        ax.set_xlim(0, 2 * np.pi)
        ax.set_ylim(-1, 1)
        return ln,

    def update(frame):
        xdata.append(frame)
        ydata.append(np.sin(frame))
        ln.set_data(xdata, ydata)
        return ln,

    ani = FuncAnimation(fig, update, frames=np.linspace(0, 2 * np.pi, 128),
                        init_func=init, blit=True)
    plt.show()


if __name__ == '__main__':
    animate_trace([])