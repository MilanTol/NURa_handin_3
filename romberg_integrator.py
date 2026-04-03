import numpy as np

# precompute 1/3
inv3 = 1 / 3


def romberg_integrator(
    func: callable, bounds: tuple, order: int = 5, err: bool = False, args: tuple = ()
) -> float:
    """
    Romberg integration method, uses extended midpoint rule to account for improper integrals
    where at one boundary func has undefined behaviour.

    Parameters
    ----------
    func : callable
        Function to integrate.
    bounds : tuple
        Lower- and upper bound for integration.
    order : int, optional
        Order of the integration: draws 2^order + 1 samples.
        The default is 5.
    err : bool, optional
        Whether to retun first error estimate.
        The default is False.
    args : tuple, optional
        Arguments to be passed to func.
        The default is ().

    Returns
    -------
    float
        Value of the integral. If err=True, returns the tuple
        (value, err), with err a first estimate of the (relative)
        error.
    """

    a, b = bounds
    h = b - a

    r = np.ndarray(
        (order,)
    )  # initialize array which will contain romberg iterations of shape (order,)
    r[0] = h * func(a + 0.5 * h, *args)  # compute initial estimate using midpoint
    N_p = (
        1  # initializes the number of points that will be sampled during each iteration
    )
    for i in range(1, order):  # ranges from 1 to m-1
        # define delta separately
        Delta = h
        # Delta must be 3*h so that you dont recompute points from previous iterations!
        h *= inv3
        # create linspace with x values in between already sampled values
        xs = np.linspace(a, b - Delta, N_p)
        # use midpoint to evaluate 2 new points for each xs.
        # note that 1.5*h sample is already included in r[i-1]
        r[i] = inv3 * (
            r[i - 1]
            + Delta * np.sum(func(xs + 0.5 * h, *args))
            + Delta * np.sum(func(xs + 2.5 * h, *args))
        )
        N_p *= 3

    # combine romberg iterations using Richardson extrapolation
    N_p = 1
    for i in range(1, order):
        N_p *= 9
        for j in range(order - i):
            r[j] = (N_p * r[j + 1] - r[j]) / (N_p - 1)

    # return best estimates
    if err:
        return r[0], np.abs(r[0] - r[1])
    else:
        return r[0]
