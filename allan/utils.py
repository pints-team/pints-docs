import numpy as np
from scipy.stats import norm, gaussian_kde
from scipy.stats import pearsonr, multivariate_normal
from scipy.integrate import quad


def get_normal_samples(num_samples, mu, sigma):
    samples = np.random.normal(scale=sigma, loc=mu, size=num_samples)
    return samples


def get_noisy_logistic_samples(num_samples, kappa, alpha, sigma, sample_min=0., sample_max=800.0):
    xs = np.arange(sample_min, sample_max, (sample_max-sample_min)/num_samples)

    orig_ys = logistic_fn(xs, kappa, alpha)
    ys = orig_ys+np.random.normal(0.0, sigma, size=xs.shape)  # Add noise
    return xs, ys


def get_gaussian_samples(num_samples, mu, sigma, sample_min=0., sample_max=800.0):
    xs = np.arange(sample_min, sample_max, (sample_max-sample_min)/num_samples)

    org_ys = gaussian_fn(xs, mu, mu)
    ys = org_ys+np.random.normal(mu, sigma, size=xs.shape)  # Add noise
    return xs, ys


def gaussian_fn(x, mu, sigma):
    return multivariate_normal.pdf(x, mean=mu, cov=sigma)


def logistic_fn(x, kappa, alpha):
    # See https://en.wikipedia.org/wiki/Logistic_function#In_ecology:_modeling_population_growth
    p_0 = 2.0  # Taken as known and constant
    return kappa / (1.0 + (kappa-p_0)/p_0*np.exp(-alpha * x))  # x either scalar or ndarray


def dlogisticdt(x, kappa, alpha, i):
    e_ax = np.exp(alpha*x)
    if i==0:  # dlogisticdkappa
        num = 4*e_ax*(e_ax-1)
        den = (2*e_ax+kappa-2)**2
        return num/den
    elif i==1:  # dlogisticdalpha
        num = 2*e_ax*x*(kappa-2)*kappa
        den = (2*e_ax+kappa-2)**2
        return num/den
    else:
        raise ValueError("i:",i)


def d2logisticdt2(x,kappa, alpha, i, j):
    e_ax = np.exp(alpha*x)
    if i==j==0:  # dkappa2
        num = -8*e_ax*(e_ax-1)
        den = (2*e_ax+kappa-2)**3
        return num/den
    if i==j==1:  # dalpha2
        num = -2*e_ax*x**2*kappa*(kappa-2)*(2*e_ax+2-kappa)
        den = (2*e_ax+kappa-2)**3
        return num/den
    num = 4*e_ax*x*(2-kappa+2*e_ax*(kappa-1))
    den = (2*e_ax+kappa-2)**3
    return num/den


def noisy_logistic_log_likelihood(sample, kappa, alpha, sigma):
    x, y = sample
    pred_y = logistic_fn(x, kappa, alpha)  # logistic function
    ret = norm.logpdf(y, pred_y, sigma)
    return ret


def get_log_unif_prior(*prior_params):
    def log_unif_prior(params):
        log_prior_prob = 0
        for param, (min_, max_) in zip(params, prior_params):
            log_prior_prob += np.log(1/(max_-min_)) if min_ <= param <= max_ else -float("inf")
        return log_prior_prob
    return log_unif_prior


def dnoisy_logistic_log_likelihood_dtheta(sample, k, a, s):
    p_0 = 2.0  # Taken as known
    x, y = sample

    # https://www.wolframalpha.com/input/?i=d%2Fdk+(ln(e%5E(-(y-k%2F(1%2B(k-2)%2F2*e%5E(-a*x)))%5E2%2F(2s%5E2))%2F(sqrt(2*pi)*s))
    e_ax = np.exp(a * x)
    dldk = -4*e_ax*(e_ax-1)*(2 * e_ax * (k - y) - (k - p_0) * y) / (s ** 2 * (2 * e_ax + k - 2) ** 3)
    # dldk2 = -2*(y-k/(1+(k-2)/2/e_ax))*-((1+(k-2)/2/e_ax)-k*1/2/e_ax)/((1+(k-2)/2/e_ax)**2)/(2*s**2)  # pen and paper
    dlda = (k-2)*k*x/e_ax*(y-k/(1/2*(k-2)/e_ax+1))/(2*s**2*(1/2*(k-2)/e_ax+1)**2)
    # dlda2 = -2*(y-k/(1+(k-2)/2/e_ax))*k*(k-2)/2/e_ax*-x/((1+(k-2)/2/e_ax)**2)/(2*s**2)  # pen and paper

    dlds = (y-k/(1+(k-2)/2/e_ax))**2/(s**3)-1/s

    if np.isnan(dldk) or np.isnan(dlda) or np.isnan(dlds):
        raise ValueError("Not a number - sample, k, a, s:",sample, k, a, s)
    dldt = np.array([dldk, dlda, dlds])
    return dldt


def get_dlogldt(dlldt_fn, data):
    def dlogldt(theta):
        sum_dlogl = np.zeros(shape=theta.shape)
        for sample in data:
            sum_dlogl += dlldt_fn(sample, *theta)  # Note POSITIVE sign
        return sum_dlogl

    return dlogldt


def get_logl(ll_fn, data):
    def logl(theta):
        sum_logl = 0
        for sample in data:
            sum_logl += ll_fn(sample, *theta)
        return sum_logl
    return logl


def get_dUdT(dlldt, data):
    """
    Note: Should have "+ dlpdt(sample, *theta)"-term, but is 0 for unif priors
    """
    def dUdT(theta):
        dudt = np.zeros(shape=theta.shape)
        for sample in data:
            dudt -= dlldt(sample, *theta)  # Note negative sign since U(T) = -loglikelihood
        return dudt
    return dUdT


def d2noisy_lll_dtheta2(sample, k, a, s, i):
    """
    i: i==0 for k, i==1 for a, i==2 for s
    """
    x,y = sample
    e_ax = np.exp(a*x)
    if i==0:
        dkdk = -(8*e_ax*(e_ax-1)*((-2*e_ax*(k-y+1))+2*(e_ax**2)+(k-2)*y))/(s**2*(2*e_ax+k-2)**4)
        dkda_sum_term = -2*k**2*y*e_ax-4*k**2*e_ax+6*k**2*e_ax**2+8*k*y*e_ax-4*k*y*e_ax**2+8*k*e_ax-8*k*e_ax**2-8*y*e_ax+4*y*e_ax**2+k**2*y-4*k*y+4*y
        dkda = -1/(s**2*(2*e_ax+k-2)**4)*4*x*e_ax*dkda_sum_term
        dkds = (8*e_ax*(e_ax-1)*(2*e_ax*(k-y)-(k-2)*y))/(s**3*(2*e_ax+k-2)**3)
        return np.array([dkdk,dkda,dkds])
    elif i==1:
        dadk_sum_term = -2*k**2*y*e_ax-4*k**2*e_ax+6*k**2*e_ax**2+8*k*y*e_ax-4*k*y*e_ax**2+8*k*e_ax-8*k*e_ax**2-8*y*e_ax+4*y*e_ax**2+k**2*y-4*k*y+4*y
        dadk = -1/(s**2*(2*e_ax+k-2)**4)*4*x*e_ax*dadk_sum_term
        dada = (2*(k-2)*k*x**2*e_ax*(4*e_ax**2*(k-y)-4*k*(k-2)*e_ax+(k-2)**2*y))/(s**2*(2*e_ax+k-2)**4)
        dads = -((k-2)*k*x/e_ax*(y-k/(1/2*(k-2)/e_ax+1)))/(s**3*(1/2*(k-2)/e_ax+1)**2)
        return np.array([dadk, dada, dads])
    elif i==2:
        dsdk = (8*e_ax*(e_ax-1)*(2*e_ax*(k-y)-(k-2)*y))/(s**3*(2*e_ax+k-2)**3)
        dsda = -((k-2)*k*x/e_ax*(y-k/(1/2*(k-2)/e_ax+1)))/(s**3*(1/2*(k-2)/e_ax+1)**2)
        dsds = 1/s**2-3*(y-k/(1/2*(k-2)/e_ax+1))**2/s**4
        return np.array([dsdk, dsda, dsds])
    else:
        raise ValueError("Only i in 0,1,2. Got", i)


def get_effective_n(chain, max_num_lags=1000, use_standard_ess=False):
    cum_autocorrs = np.zeros(chain.shape[1])
    max_num_lags = min(max_num_lags, chain.shape[1]-1)
    for i in range(chain.shape[1]):
        lag = 0
        cum_autocorr = 0
        prev_auto_corr = 1
        while True:
            lag += 1
            auto_corr,_ = pearsonr(chain[:-lag, i], chain[lag:, i])
            if not use_standard_ess and prev_auto_corr + auto_corr < 0:
                break
            if use_standard_ess and auto_corr < 0:
                break
            cum_autocorr += auto_corr
            if lag == max_num_lags:
                break
            prev_auto_corr = auto_corr
        cum_autocorrs[i] = cum_autocorr
    return chain.shape[0]/(1+2*cum_autocorrs)


def get_kdes_of_chain(chain):
    kdes = []
    for i in range(chain.shape[1]):
        trace = chain[:, i]
        kde = gaussian_kde(trace)
        kdes.append(kde)
    return kdes


def get_kl_div_for_univariate_pdfs(p, q, min_x=None, max_x=None):
    kl_fn = lambda x: p(x)*np.log((p(x))/(q(x)))
    min_x = -np.inf if min_x is None else min_x
    max_x = np.inf if max_x is None else max_x
    kl_div, kl_div_err_approx = quad(kl_fn, a=min_x, b=max_x)
    return kl_div, kl_div_err_approx


def get_deriv_of_fim_w_gauss_assump(dtargetfndt, d2targetfndt2, data, deriv_logprior_hessian=None):
    def deriv_fim(theta, sigma):
        sens = np.zeros(shape=(theta.size, len(data)))
        dsens = np.zeros(shape=(theta.size, theta.size, len(data)))
        dG = np.zeros(shape=(theta.size, theta.size, theta.size))

        for i in range(theta.size):
            for sample_i, sample in enumerate(data):
                x, _ = sample
                sens[(i, sample_i)] = dtargetfndt(x, *theta, i)
                for k in range(theta.size):
                    dsens[(i, k, sample_i)] = d2targetfndt2(x, *theta, i, k)

        for i in range(theta.size):
            for j in range(theta.size):
                for k in range(theta.size):
                    dG[(i, j, k)] = (np.dot(dsens[i,k],sens[j])+np.dot(sens[i],dsens[j,k]))/sigma**2
                    if deriv_logprior_hessian is not None:
                        dG[(i,j,k)] += deriv_logprior_hessian[(i,j,k)]  # TODO
        return dG
    return deriv_fim


def get_fim_w_gauss_assump(dtargetfndt, data, logprior_hessian=None):
    def fim(theta, sigma):  # theta: Vec of params w/ gaussian noise sigma as -1st param
        sens = np.zeros(shape=(theta.size, len(data)))
        G = np.zeros(shape=(theta.size,theta.size))
        for i in range(theta.size):
            for j, sample in enumerate(data):
                x, _ = sample
                sens[(i,j)] = dtargetfndt(x, *theta, i)

        for i in range(theta.size):
            for j in range(theta.size):
                G[(i, j)] = np.dot(sens[i],sens[j])/sigma**2
                if logprior_hessian is not None:
                    G[(i, j)] += logprior_hessian[(i,j)]  # TODO

        return G
    return fim
