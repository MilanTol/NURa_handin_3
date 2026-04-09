import numpy as np


def chi2(
    model: callable,
    y_data: np.ndarray,
    x_data: np.ndarray,
    err_data: np.ndarray,
    args: np.ndarray = np.array(()),
) -> float:
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
    # we compute this with a for loop since were using integration in our code
    # the code implemented for integration doesnt handle arrays well.
    res = 0
    for i in range(len(x_data)):
        model_val = model(x_data[i], *args)
        res += (
            (y_data[i]-model_val) * (y_data[i]-model_val)
            / (err_data[i] * err_data[i])
        )
    return res
