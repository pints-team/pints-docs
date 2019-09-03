from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints.toy as toy
import pints
import numpy as np
import logging
import math
import sys
from numpy import inf
import copy 
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.cm as cm
import matplotlib.transforms as transforms
import copy

model_dict = dict()

# LOGISTIC MODEL # 
log_dict = dict()

log_dict['name'] = "Logistic model"
log_dict['model'] = toy.LogisticModel()

log_dict['real_parameters'] = [0.5, 500]
log_dict['times']  = log_dict['model'].suggested_times()

log_dict['boundaries'] = pints.RectangularBoundaries([0, 200], [1, 1000])

# Choose an initial position
log_dict['x0'] = [0.4, 420]

model_dict['Logistic'] = log_dict


# ROSENBROCK Distribution # 
mod_dict = dict()

mod_dict['name'] = "ROSENBROCK"

log_pdf = pints.toy.RosenbrockLogPDF()
mod_dict['model'] = log_pdf
# Use suggested prior bounds
bounds = log_pdf.suggested_bounds()
# Create a uniform prior over both the parameters
log_prior = pints.UniformLogPrior(
    bounds[0], bounds[1])
# Create a posterior log-likelihood (log(likelihood * prior))
log_posterior = pints.LogPosterior(log_pdf, log_prior)
mod_dict['score'] = log_posterior

mod_dict['real_parameters'] = [7, 15]
mod_dict['times']  = []

# Choose an initial position
mod_dict['x0'] = [0.5, 3]

model_dict['Rosenbrock'] = mod_dict


def cov_ellipse(cov, mean, ls, cov_color, ax):
    
    n_std = 2

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of a
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        color=cov_color, 
        ls=ls)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    ellipse.set_facecolor('none') 
    
    return ellipse

def add_contour(x_max, x_min, y_max, y_min, score, levels, is_dist=False):
    x0 = np.linspace(x_min, x_max, 200)
    x1 = np.linspace(y_min, y_max, 200)
    X0, X1 = np.meshgrid(x0,x1)
    if is_dist:
        Y = np.exp([[score([i, j]) for i in x0] for j in x1])
    else:
        Y = np.array([ [score([X0[i][j], X1[i][j]]) for j in range(len(X0[0]))] for i in range(len(X0))])
    cp = plt.contour(X0, X1, Y, levels = levels)
    return cp
    

def draw_change(prev_mean,prev_cov, opt, score, change, lims=None):
    cov_color = 'black'    
    
    mean = opt.mean() if 'mean' in change else prev_mean
    cov = opt.cov() if 'cov' in change else (
                        opt.rankmu() if 'rankmu' in change else prev_cov)
   
    ax = plt.subplot()
    if lims:
        x_max, x_min, y_max, y_min = lims
    else:
        x_max = 2.
        x_min = -.5
        y_max = max([mean[1], prev_mean[1]]) + 5
        y_min = min([mean[1], prev_mean[1]]) - 5
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    
    ell_updated = cov_ellipse(cov, mean, '-', cov_color, ax)
    ax.add_artist(ell_updated)
    
    ell_old = cov_ellipse(prev_cov, prev_mean, '--', cov_color, ax)
    ax.add_artist(ell_old)
    
    ax.scatter([mean[0]], [mean[1]], facecolor='black', color='black')
    ax.scatter([prev_mean[0]], [prev_mean[1]],facecolors="None", color='black')
    
    levels = [200.0, 400.0, 800.0, 1600.0, 3600.0, 7200.0, 14400.0, 28800.0]
    cp = add_contour(x_max, x_min, y_max, y_min, score, levels)
    ax.clabel(cp, inline=1, fontsize=10, fmt='%1.0f')
    
    plt.show()
    
    
def draw_samples(fxs, xs, opt, score, lims=None):
    best_color = 'green'
    worst_color = 'red'
    cov_color = 'black'
    point_area = 10

    order = np.argsort(fxs)
    xs_bests = np.array(xs[order])

    ax = plt.subplot()
    if lims:
        x_max, x_min, y_max, y_min = lims
    else:
        x_max = max(xs_bests[:,0]) + 0.1
        x_min = min(xs_bests[:,0]) - 0.1
        y_max = max(xs_bests[:,1]) + 5
        y_min = min(xs_bests[:,1]) - 5
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    
    cov = opt.cov()
    mean = opt.mean()

    ell = cov_ellipse(cov, mean, '-', cov_color, ax)
    ax.add_artist(ell)

    half = len(xs_bests) // 2 
    X_best = xs_bests[:half,0]
    Y_best = xs_bests[:half,1]
    for i in range(len(X_best)):
        ax.plot([mean[0], X_best[i]], [mean[1], Y_best[i]], color=best_color)
    ax.scatter(X_best,Y_best, facecolor=best_color, color=best_color, s=point_area)

    X_worst = xs_bests[half:,0]
    Y_worst = xs_bests[half:,1]
    for i in range(len(X_worst)):
        ax.plot([mean[0], X_worst[i]], [mean[1], Y_worst[i]], color=worst_color)
    ax.scatter(X_worst ,Y_worst, facecolor=worst_color, color=worst_color, s=point_area)
    
    levels = [200.0, 400.0, 800.0, 1600.0, 3600.0, 7200.0, 14400.0, 28800.0]
    cp = add_contour(x_max, x_min, y_max, y_min, score, levels)
    ax.clabel(cp, inline=1, fontsize=10, fmt='%1.0f')

    plt.show()
    
def draw_samples_weights(fxs, xs, opt, score, n_best, 
                         lims=None, fakeIGO=False, 
                         levels=[], log_pdf=None,
                         name=None):
    best_color = 'green'
    worst_color = 'red'
    cov_color = 'black'
    point_area = 15
    order = np.argsort(fxs)
    xs_bests = np.array(xs[order])

    ax = plt.subplot()
    if lims:
        x_max, x_min, y_max, y_min = lims
    else:
        x_max = max(xs_bests[:,0]) + 0.1
        x_min = min(xs_bests[:,0]) - 0.1
        y_max = max(xs_bests[:,1]) + 1
        y_min = min(xs_bests[:,1]) - 1
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    plt.xlabel('x0')
    plt.ylabel('x1')
    
    cov = opt.cov()
    mean = opt.mean()
    try:
        sigma = opt.sigma()
        ell = cov_ellipse(sigma**2 * cov, mean, '-', cov_color, ax)
    except:
        ell = cov_ellipse(cov, mean, '-', cov_color, ax)
    ax.add_artist(ell)

    if fakeIGO:
        n_best = len(xs_bests) // 2
    X_best = xs_bests[:n_best,0]
    Y_best = xs_bests[:n_best,1]
    if fakeIGO:
        best_color = cm.brg(np.linspace(.9, .75, len(X_best)))
    
    ax.scatter(X_best,Y_best, facecolor=best_color, color=best_color, s=point_area)

    X_worst = xs_bests[n_best:,0]
    Y_worst = xs_bests[n_best:,1]
    ax.scatter(X_worst ,Y_worst, facecolor=worst_color, color=worst_color, s=point_area)
    if len(levels) == 0:
        levels = [200.0, 400.0, 800.0, 1600.0, 3600.0, 7200.0, 14400.0, 28800.0]
    if log_pdf:
        cp = add_contour(x_max, x_min, y_max, y_min, log_pdf, levels, is_dist=True)
        ax.clabel(cp, inline=1, fontsize=10, fmt='%1.2f')

    else:
        cp = add_contour(x_max, x_min, y_max, y_min, score, levels)
        ax.clabel(cp, inline=1, fontsize=10, fmt='%1.0f')

    if name:
        plt.savefig(name + "samples.pdf")
        
    plt.show()
    
    
# ONLY USABLE ON LOGISTIC MODEL    
def draw_samples_weights_boundaries(fxs, xs, opt, score, n_best, boundaries, name=None):
    best_color = 'green'
    worst_color = 'red'
    cov_color = 'black'
    bound_color = 'black'
    point_area = 15

    order = np.argsort(fxs)
    xs_bests = np.array(xs[order])

    ax = plt.subplot()

    x_max = 3.0
    x_min = -0.1
    y_max = 430
    y_min = 415
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    plt.xlabel('x0')
    plt.ylabel('x1')
    
    cov = opt.cov()
    mean = opt.mean()

    ell = cov_ellipse(cov, mean, '-', cov_color, ax)
    ax.add_artist(ell)

    X_best = xs_bests[:n_best,0]
    Y_best = xs_bests[:n_best,1]
    
    ax.scatter(X_best,Y_best, facecolor=best_color, color=best_color, s=point_area)

    X_worst = xs_bests[n_best:,0]
    Y_worst = xs_bests[n_best:,1]
    ax.scatter(X_worst ,Y_worst, facecolor=worst_color, color=worst_color, s=point_area)
    
    boundaries_upper = boundaries.upper()
    boundaries_lower = boundaries.lower()
    ax.plot([boundaries_lower[0], boundaries_upper[0]], [boundaries_lower[1], boundaries_lower[1]], color=bound_color, linestyle='dashed')
    ax.plot([boundaries_lower[0], boundaries_upper[0]], [boundaries_upper[1], boundaries_upper[1]], color=bound_color, linestyle='dashed')
    ax.plot([boundaries_lower[0], boundaries_lower[0]], [boundaries_lower[1], boundaries_upper[1]], color=bound_color, linestyle='dashed')
    ax.plot([boundaries_upper[0], boundaries_upper[0]], [boundaries_lower[1], boundaries_upper[1]], color=bound_color, linestyle='dashed')
    
    
    levels = [200.0, 400.0, 800.0, 1600.0, 3600.0, 7200.0, 14400.0, 28800.0]
    cp = add_contour(x_max, x_min, y_max, y_min, score, levels)
    ax.clabel(cp, inline=1, fontsize=10, fmt='%1.0f')

    if name:
        plt.savefig(name + "boundaries.pdf")
    
    plt.show()


def draw_means(means, score, levels=[], is_dist=False, lims=[], name=None):
    line_color = 'blue'
    
    if len(lims) ==0:
        x_max = max(means[:][0]) + 0.5
        x_min = min(means[:][0]) - 2
        y_max = max(means[:][1]) + 2
        y_min = min(means[:][1]) - 2
    else:
        x_max, x_min, y_max, y_min = lims
        
    ax = plt.subplot()
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    plt.xlabel('x0')
    plt.ylabel('x1')
    # If we plot a the means of multiple runs
    if isinstance(means[0], list):
        for one_run_means in means:
            for i in range(len(one_run_means) - 1):
                mean = one_run_means[i]
                next_mean = one_run_means[i+1]
                ax.plot([mean[0], next_mean[0]], [mean[1], next_mean[1]], color=line_color)   
    
    # If we plot a the means of a single runs
    else:
        for i in range(len(means) - 1):
            mean = means[i]
            next_mean = means[i+1]
            ax.plot([mean[0], next_mean[0]], [mean[1], next_mean[1]], color=line_color)    
            
    if len(levels) == 0:
        levels = [1.0, 4.5, 5.0, 10.0, 200.0, 400.0, 800.0, 1600.0, 3600.0, 7200.0, 14400.0, 28800.0]
    cp = add_contour(x_max, x_min, y_max, y_min, score, levels, is_dist=is_dist)
    if is_dist:
        ax.clabel(cp, inline=1, fontsize=10, fmt='%1.2f')
    else:
        ax.clabel(cp, inline=1, fontsize=10, fmt='%1.0f')

    if name:
        plt.savefig(name + "mean.pdf")
    plt.show()


    
def draw_cma_evolution(model,
              real_parameters,
              times,
              opt,
              n_best,         
              fakeIGO=False,
              iterations=500,
              name=None):
    
    values, _ = model.simulateS1(real_parameters, times)
    values += np.random.normal(0, 10, values.shape)
    my_problem = pints.SingleOutputProblem(model, times, values)
    score = pints.MeanSquaredError(my_problem)
    
    means = []
    lims = [1,0,530,415]
    try:
        for i in range(iterations):
            xs = opt.ask()
            means.append(opt.mean())
            fxs = [score(x) for x in xs]
            if i== 0 or i== iterations - 1:
                if name:
                    draw_samples_weights(fxs, xs, opt, score, n_best,  lims=lims,
                                 fakeIGO=fakeIGO, name=name + "_" +str(i))
                else:
                    draw_samples_weights(fxs, xs, opt, score, n_best,  lims=lims,
                                 fakeIGO=fakeIGO)
            #else:
             #   draw_samples_weights(fxs, xs, opt, score, n_best,  lims=lims,
                                # fakeIGO=fakeIGO, levels=levels, log_pdf=log_pdf)
            
            opt.tell(fxs)
            if i== 0 or i== iterations - 1:
                print("Iteration #",i)
                print(math.exp(log_pdf(opt.xbest())))
    except:
        print("Error occured")
    draw_means(means, score, lims=lims, name=name)

def draw_dist_model_evolution(log_pdf,
              opt,
              score,
              n_best,
              fakeIGO=False,
              iterations=500,
              name=None): 
    
    means = []
    levels = np.concatenate((np.linspace(0, 0.05, 3), np.linspace(0.1, 1, 5))) 
    lims = [2,-1,5,-1]
    try:
        for i in range(iterations):
            xs = opt.ask()
            means.append(opt.mean())
            fxs = [- score(x) for x in xs]
            if i== 0 or i== iterations - 1:
                if name:
                    draw_samples_weights(fxs, xs, opt, score, n_best,  lims=lims,
                                 fakeIGO=fakeIGO, levels=levels, log_pdf=log_pdf, name=name + "_" +str(i))
                else:
                    draw_samples_weights(fxs, xs, opt, score, n_best,  lims=lims,
                                 fakeIGO=fakeIGO, levels=levels, log_pdf=log_pdf)
                
            
            #else:
             #   draw_samples_weights(fxs, xs, opt, score, n_best,  lims=lims,
                                # fakeIGO=fakeIGO, levels=levels, log_pdf=log_pdf)
            
            opt.tell(fxs)
            if i== 0 or i== iterations - 1:
                print("Iteration #",i)
                print(math.exp(log_pdf(opt.xbest())))
    except:
        print("Error occured")
    draw_means(means, log_pdf, levels=levels, is_dist=True, lims=lims, name=name)

    
def draw_means_path(model,
              real_parameters,
              times,
              og_opt,
              iterations=500,
              n_runs=20):
    
    values, _ = model.simulateS1(real_parameters, times)
    values += np.random.normal(0, 10, values.shape)
    my_problem = pints.SingleOutputProblem(model, times, values)
    score = pints.MeanSquaredError(my_problem)
    lims = [1,0,530,415]
    all_means = []
    
    for _ in range(n_runs):
        opt = copy.deepcopy(og_opt)
        means = []
        try:
            for i in range(iterations):
                xs = opt.ask()
                means.append(opt.mean())
                fxs = [score(x) for x in xs]
                opt.tell(fxs)
        except:
            continue
        all_means.append(means)
    draw_means(all_means, score, lims=lims)
    
def draw_means_path_dist_model(log_pdf,
              og_opt,
              score,
              iterations=500,
              n_runs=20):
    
    lims = [1,0,530,415]
    all_means = []
    
    for _ in range(n_runs):
        opt = copy.deepcopy(og_opt)
        means = []
        try:
            for i in range(iterations):
                xs = opt.ask()
                means.append(opt.mean())
                fxs = [ - score(x) for x in xs]
                opt.tell(fxs)
        except:
            continue
        all_means.append(means)
    draw_means(all_means, log_pdf, lims=lims)
        
def draw_boundaries(model,
              real_parameters,
              times,
              opt_bound, 
              opt, 
              n_best, 
              boundaries, 
              name):
    
    values, _ = model.simulateS1(real_parameters, times)
    values += np.random.normal(0, 10, values.shape)
    my_problem = pints.SingleOutputProblem(model, times, values)
    score = pints.MeanSquaredError(my_problem)
    
    xs = opt.ask()
    fxs = [score(x) for x in xs]
    draw_samples_weights_boundaries(fxs, xs, opt, score, n_best, boundaries, name=name+"_1")
    
    xs = opt_bound.ask()
    fxs = [score(x) for x in xs]
    draw_samples_weights_boundaries(fxs, xs, opt, score, n_best, boundaries,  name=name+"_2")
   