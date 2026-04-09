import numpy as np


def gradient(
    func: callable, args: np.ndarray, h: float = 1e-2, min_stepsize: float = 1e-10
) -> np.ndarray:
    """
    Computes partial derivative of func wrt each argument.
    Differentiation is done using finite difference.

    Args:
        func (callable):
            func that takes args as input
        args (np.ndarray):
            arguments of callable, arguments must be primitive types (floats, ints)!
        h (float, optional):
            factor of current argument value that is used as stepsize for finite difference
        min_stepsize (float, optional):
            minimum stepsize to take in case the argument value is 0

    Returns:
        gradient (np.ndarray):
            partial derivatives of function. k'th index corresponds to
            del func / del p_k
    """

    n = len(args)  # store number of arguments as n
    derivs = np.ndarray(n, dtype=float)  # instantiate array of size n

    for k in range(n):  # loop over each argument

        # copy the argument arrays to not alter any values:
        # since args only contains primitive types we use in-built shallow copy of np.ndarray
        args_ph, args_mh = args.astype(float).copy(), args.astype(float).copy()

        # compute stepsize, scaled accordingly to args size.
        step = max(h * np.abs(args[k]), min_stepsize)
        args_ph[k] += step  # apply stepsizes
        args_mh[k] -= step

        # compute partial derivative using finite difference:
        derivs[k] = 1 / (2 * step) * (func(args_ph) - func(args_mh))

    return derivs
