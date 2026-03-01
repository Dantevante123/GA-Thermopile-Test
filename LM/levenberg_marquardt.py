# Code adapted from Gavin, H.P. (2020) The Levenberg-Marquardt method for 
# nonlinear least squares curve-fitting problems.
# https://people.duke.edu/~hpgavin/ce281/lm.pdf

import numpy as np
#import seaborn as sns
import matplotlib.pylab as pl
import matplotlib.pyplot as plt

def lm_func(t, p):
    p = np.asarray(p, dtype=float).reshape(-1)
    V0, K, n = float(p[0]), float(p[1]), float(p[2])

    t = np.asarray(t, dtype=float)

    if t.ndim == 2 and t.shape[1] == 2:
        T = t[:, 0]
        Tbg = t[:, 1]
    else:
        T = t.reshape(-1)
        Tbg = 0.0

    return V0 + K * (T**n - Tbg**n)


def lm_FD_J(t,p,y,dp):
    """

    Computes partial derivates (Jacobian) dy/dp via finite differences.

    Parameters
    ----------
    t  :     independent variables used as arg to lm_func (m x 1) 
    p  :     current parameter values (n x 1)
    y  :     func(t,p,c) initialised by user before each call to lm_FD_J (m x 1)
    dp :     fractional increment of p for numerical derivatives
                - dp(j)>0 central differences calculated
                - dp(j)<0 one sided differences calculated
                - dp(j)=0 sets corresponding partials to zero; i.e. holds p(j) fixed

    Returns
    -------
    J :      Jacobian Matrix (n x m)

    """

    global func_calls
    
    # number of data points
    m = len(y)
    # number of parameters
    n = len(p)

    # initialize Jacobian to Zero
    ps=p.copy()
    J=np.zeros((m,n)) 
    del_=np.zeros((n,1))
    
    # START --- loop over all parameters
    for j in range(n):
        # parameter perturbation
        del_[j,0] = dp[j,0] * (1+abs(p[j,0]))
        # perturb parameter p(j)
        p[j,0]   = ps[j,0] + del_[j,0]
        
        if del_[j,0] != 0:
            y1 = lm_func(t,p)
            func_calls = func_calls + 1
            
            if dp[j,0] < 0: 
                # backwards difference
                J[:,j] = (y1-y)/del_[j,0]
            else:
                # central difference, additional func call
                p[j,0] = ps[j,0] - del_[j]
                J[:,j] = (y1-lm_func(t,p)) / (2 * del_[j,0])
                func_calls = func_calls + 1
        
        # restore p(j)
        p[j,0]=ps[j,0]
        
    return J
    

def lm_Broyden_J(p_old,y_old,J,p,y):
    """
    Carry out a rank-1 update to the Jacobian matrix using Broyden's equation.

    Parameters
    ----------
    p_old :     previous set of parameters (n x 1)
    y_old :     model evaluation at previous set of parameters, y_hat(t,p_old) (m x 1)
    J     :     current version of the Jacobian matrix (m x n)
    p     :     current set of parameters (n x 1)
    y     :     model evaluation at current  set of parameters, y_hat(t,p) (m x 1)

    Returns
    -------
    J     :     rank-1 update to Jacobian Matrix J(i,j)=dy(i)/dp(j) (m x n)

    """
    
    h = p - p_old
    
    a = (np.array([y - y_old]).T - J@h)@h.T
    b = h.T@h

    # Broyden rank-1 update eq'n
    J = J + a/b

    return J

def lm_matx(t,p_old,y_old,dX2,J,p,y_dat,weight,dp):
    """
    Evaluate the linearized fitting matrix, JtWJ, and vector JtWdy, and 
    calculate the Chi-squared error function, Chi_sq used by Levenberg-Marquardt 
    algorithm (lm).
    
    Parameters
    ----------
    t      :     independent variables used as arg to lm_func (m x 1)
    p_old  :     previous parameter values (n x 1)
    y_old  :     previous model ... y_old = y_hat(t,p_old) (m x 1)
    dX2    :     previous change in Chi-squared criteria (1 x 1)
    J      :     Jacobian of model, y_hat, with respect to parameters, p (m x n)
    p      :     current parameter values (n x 1)
    y_dat  :     data to be fit by func(t,p,c) (m x 1)
    weight :     the weighting vector for least squares fit inverse of 
                 the squared standard measurement errors
    dp     :     fractional increment of 'p' for numerical derivatives
                  - dp(j)>0 central differences calculated
                  - dp(j)<0 one sided differences calculated
                  - dp(j)=0 sets corresponding partials to zero; i.e. holds p(j) fixed

    Returns
    -------
    JtWJ   :     linearized Hessian matrix (inverse of covariance matrix) (n x n)
    JtWdy  :     linearized fitting vector (n x m)
    Chi_sq :     Chi-squared criteria: weighted sum of the squared residuals WSSR
    y_hat  :     model evaluated with parameters 'p' (m x 1)
    J :          Jacobian of model, y_hat, with respect to parameters, p (m x n)

    """
    
    global iteration,func_calls
    
    # number of parameters
    Npar   = len(p)

    # evaluate model using parameters 'p'
    y_hat = lm_func(t,p)
    
    func_calls = func_calls + 1

    if not np.remainder(iteration,2*Npar) or dX2 > 0:
        # finite difference
        J = lm_FD_J(t,p,y_hat,dp)
    else:
        # rank-1 update
        J = lm_Broyden_J(p_old,y_old,J,p,y_hat)

    # residual error between model and data
    delta_y = np.array([y_dat - y_hat]).T
    
    # Chi-squared error criteria
    Chi_sq = (delta_y.T @ (delta_y * weight)).item()
 
    JtWJ  = J.T @ ( J * ( weight * np.ones((1,Npar)) ) )
    
    JtWdy = J.T @ ( weight * delta_y )
    
    
    return JtWJ,JtWdy,Chi_sq,y_hat,J

def lm(p, t, y_dat):
    """
    Levenberg–Marquardt curve-fitting
    Supports both 1D t and 2D t = [T, Tbg]
    """

    global iteration, func_calls

    # --- säkra former ---
    t = np.asarray(t, dtype=float)
    y_dat = np.asarray(y_dat, dtype=float).reshape(-1)

    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        p = p.reshape(-1, 1)

    iteration = 0
    func_calls = 0
    eps = 2 ** (-52)

    Npar = p.shape[0]
    Npnt = y_dat.size

    # length check
    if t.ndim == 2:
        if t.shape[0] != y_dat.size:
            raise ValueError("The length of t must equal the length of y_dat!")
    else:
        t = t.reshape(-1)
        if t.size != y_dat.size:
            raise ValueError("The length of t must equal the length of y_dat!")

    # initial values
    p_old = np.zeros((Npar, 1))
    y_old = np.zeros((Npnt, 1))

    X2 = float(1e-3 / eps)
    X2_old = float(1e-3 / eps)

    J = np.zeros((Npnt, Npar))
    DoF = float(Npnt - Npar + 1)

    # weights
    denom = float(np.dot(y_dat, y_dat))
    if denom <= 0:
        denom = 1e-12
    weight_vec = (1.0 / denom) * np.ones((Npnt, 1))

    dp = -0.001 * np.ones((Npar, 1))
    p_min = -100 * np.abs(p)
    p_max = 100 * np.abs(p)

    MaxIter = 1000
    epsilon_1 = 1e-3
    epsilon_2 = 1e-3
    epsilon_4 = 1e-1
    lambda_ = 1e-2
    lambda_UP_fac = 11
    lambda_DN_fac = 9

    idx = np.arange(Npar)
    stop = 0

    JtWJ, JtWdy, X2, y_hat, J = lm_matx(
        t, p_old, y_old, 1, J, p, y_dat, weight_vec, dp
    )

    cvg_hst = np.ones((MaxIter, Npar + 2))

    while not stop and iteration < MaxIter:
        iteration += 1

        A = JtWJ + lambda_ * np.diag(np.diag(JtWJ))
        h = np.linalg.solve(A, JtWdy)

        p_try = p + h[idx]
        p_try = np.minimum(np.maximum(p_min, p_try), p_max)

        delta_y = np.array([y_dat - lm_func(t, p_try)]).T
        func_calls += 1

        X2_try = float((delta_y.T @ (delta_y * weight_vec)).item())
        den = X2 - X2_try
        if abs(den) < 1e-18:
            den = 1e-18

        num = float((h.T @ (lambda_ * h + JtWdy)).item())
        rho = num / den

        if rho > epsilon_4:
            X2_old = X2
            p_old = p
            y_old = y_hat
            p = p_try

            JtWJ, JtWdy, X2, y_hat, J = lm_matx(
                t, p_old, y_old, X2 - X2_old, J, p, y_dat, weight_vec, dp
            )

            lambda_ = max(lambda_ / lambda_DN_fac, 1e-7)
        else:
            lambda_ = min(lambda_ * lambda_UP_fac, 1e7)

        cvg_hst[iteration - 1, 0] = func_calls
        cvg_hst[iteration - 1, 1] = float(X2 / DoF)
        for i in range(Npar):
            cvg_hst[iteration - 1, i + 2] = float(p[i, 0])

        if np.max(np.abs(h) / (np.abs(p) + 1e-12)) < epsilon_2:
            stop = 1

    cvg_hst = cvg_hst[:iteration, :]
    redX2 = float(X2 / DoF)

    return p, redX2, None, None, None, None, cvg_hst



def make_lm_plots(x,y,cvg_hst):
    

    # extract parameters data
    p_hst  = cvg_hst[:,2:]
    p_fit  = p_hst[-1,:]
    y_fit = lm_func(x,np.array([p_fit]).T)
    
    # define fonts used for plotting
    font_axes = {'family': 'serif',
            'weight': 'normal',
            'size': 12}
    font_title = {'family': 'serif',
                  'weight': 'normal',
            'size': 14}       
    
    # define colors and markers used for plotting
    n = len(p_fit)
    colors = pl.cm.ocean(np.linspace(0,.75,n))
    markers = ['o','s','D','v']    
    
    # create plot of raw data and fitted curve
    fig1, ax1 = plt.subplots()
    ax1.plot(x,y,'wo',markeredgecolor='black',label='Raw data')
    ax1.plot(x,y_fit,'r--',label='Fitted curve',linewidth=2)
    ax1.set_xlabel('t',fontdict=font_axes)
    ax1.set_ylabel('y(t)',fontdict=font_axes)
    ax1.set_title('Data fitting',fontdict=font_title)
    ax1.legend()
    
    # create plot showing convergence of parameters
    fig2, ax2 = plt.subplots()
    for i in range(n):
        ax2.plot(cvg_hst[:,0],p_hst[:,i]/p_hst[0,i],color=colors[i],marker=markers[i],
                 linestyle='-',markeredgecolor='black',label='p'+'${_%i}$'%(i+1))
    ax2.set_xlabel('Function calls',fontdict=font_axes)
    ax2.set_ylabel('Values (norm.)',fontdict=font_axes)
    ax2.set_title('Convergence of parameters',fontdict=font_title) 
    ax2.legend()
    
    # create plot showing histogram of residuals
    fig3, ax3 = plt.subplots()
    ax3.hist((y_fit - y).ravel(), bins=30)
    ax3.set_xlabel('Residual error',fontdict=font_axes)
    ax3.set_ylabel('Frequency',fontdict=font_axes)
    ax3.set_title('Histogram of residuals',fontdict=font_title)
    
