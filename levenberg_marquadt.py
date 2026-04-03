import numpy as np
from chi2 import chi2
from gradient import gradient
from matrix import Matrix

def levenberg_marquardt(
    model:callable, y_data:np.ndarray, x_data:np.ndarray, err_data:np.ndarray, 
    p_init:np.ndarray, lmbda_init:float=1e-3, w=10, maxit:int=1000, rel_tol:float = 1e-4
)->np.ndarray:  
    """
    levenberg-marquardt algorithm to find parameters that minimize 
    chi^2 for the given data and model.
    This algorithm transitions from steepest descent to newtons method.

    Args:
        model (callable): 
            model of which to find optimal parameters
        y_data (np.ndarray): 
            y_values of data.
        x_data (np.ndarray): 
            x_values of data.
        err_data (np.ndarray): 
            errors on y_values of data.
        p_init (np.ndarray): 
            initial estimate of optimal parameters.
        lmbda_init (float, optional): 
            initial lambda_value, sets the step value for updates. Defaults to 1e-3.
        w (int, optional): 
            sets how quickly lmbda updates. Defaults to 10.
        maxit (int, optional): 
            maximum iterations the algorithm may take. Defaults to 1000.
        reltol (float, optional):
            relative tolerance (if chi_new - chi < reltol*chi algorithm terminates).
            Defaults to 1e-8.

    Returns:
        p (np.ndarray): 
            array containing optimal parameters.
        min_Chi2 (int):
            minimum value of Chi2 found.
    """
    
    lmbda = lmbda_init
    p = p_init.copy()
    chi = chi2(model, y_data, x_data, err_data, p)
    # precompute 1/w since we will be using this multiple times:
    w_inv = 1/w

    for j in range(maxit):
        # create chi function that only depends on model parameters:
        chi_temp = lambda args: chi2(model, y_data, x_data, err_data, args)
        # compute the gradient of chi_temp at p and compute beta
        beta = -0.5*gradient(chi_temp, p)

        # initialize alpha matrix of size (params, params,):
        alpha = np.zeros( (len(p), len(p),), dtype=float)
        # loop over data points:
        for i in range(len(x_data)):
            # create model function that only depends on model parameters:
            model_temp = lambda args: model(x_data[i], *args)
            # compute the gradient of model_temp at p
            model_gradient = gradient(model_temp, p)
            alpha += np.outer(model_gradient, model_gradient) / (err_data[i]*err_data[i])
        
        # add lmbda*(diagonal of alpha), this is what makes it the levenberg algorithm
        alpha += lmbda * np.diag(np.diag(alpha)) # np.diag(np.diag()) extracts the diagonal as a matrix

        # make alpha a matrix and solve alpha@delp = beta:
        alpha = Matrix(alpha)
        delp = alpha.solve(beta)

        # compute the chi2 value at the proposed point
        chi_new = chi2(model, y_data, x_data, err_data, p+delp)
        # see whether to approve next step
        Delta_chi = chi_new - chi
        if Delta_chi < 0: # if new point is better:
            # check termination condition:
            if np.abs(Delta_chi) < rel_tol*chi: # chi - chi_new > 0
                return p + delp, chi_new
            # otherwise update step
            p += delp 
            chi = chi_new
            lmbda *= w_inv # become more like newton's method
        else: # if new point is worse: become more like steepest descent
            lmbda *= w

    print("WARNING (levenberg-marquardt): desired tolerance not reached")
    return p, chi_new


    