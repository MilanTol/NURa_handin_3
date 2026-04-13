import numpy as np


def chi2(
        model:callable, y_data:np.ndarray, x_data:np.ndarray, err_data:np.ndarray, args:np.ndarray=np.array(())
    )->float:
    """
    returns chi^2 for a model and data

    Args:
        model (callable): 
            function that takes in x_value and potential arguments
        y_data (np.ndarray): 
            y_values of data
        x_data (np.ndarray): 
            x_values of data
        err_data (np.ndarray): 
            error in y for each data point
        args (tuple, optional): 
            arguments of the model, Defaults to ().

    Returns:
        float: chi^2 value
    """
    return np.sum(
        (y_data - model(x_data, *args))*(y_data - model(x_data, *args))
        /
        (err_data*err_data)
    )
