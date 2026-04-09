import numpy as np
from sorter import Sorter


def downhill_simplex(
    func: callable, x_init: np.ndarray, relerr: float = 1e-4, maxit: int = 1000
):
    """
    finds the optimal parameters that minimize func, using downhill-simplex
    algorithm.

    Args:
        func (callable):
            function to minimize
        x_init (np.ndarray):
            initial array of parametrizations
        relerr (float, optional):
            maximum relative error. Defaults to 1e-4.
        maxit (int, optional):
            maximum iterations before returning. Defaults to 1000.

    Returns:
        p_opt:
            the parametrization that minimizes func.
    """
    N = len(x_init) - 1
    N_inv = 1 / N
    x_points = x_init.copy()

    # compute y_values corresponding to the initial x_values
    y_vals = []
    for x_point in x_points:
        y_vals.append(func(x_point))
    y_vals = np.array(y_vals)

    for i in range(maxit):

        # sort the y_init values:
        y_vals, indx = Sorter.quicksort(None, y_vals, make_indx=True)
        # sort the x_values according to their associated y_values
        x_points = x_points[indx]

        # compute the centroid of all points except the worst.
        centroid = N_inv * np.sum(x_points[:-1], axis=0)  # exclude the worst point!

        # check whether target accuracy is reached:
        accuracy = 2 * np.abs(y_vals[0] - y_vals[-1]) / np.abs(y_vals[0] + y_vals[-1])
        if accuracy < relerr:
            return x_points[0]

        # propose new point, by flipping mirroring worst point through centroid
        x_new = 2 * centroid - x_points[-1]
        y_new = func(x_new)

        # if the new point is "decent":
        # overwrite the worst point with it.
        if y_vals[0] <= y_new < y_vals[-1]:
            x_points[-1] = x_new
            y_vals[-1] = y_new

        # if the new point is the best so far:
        elif y_new < y_vals[0]:
            # explore further in that direction:
            x_exp = 2 * x_new - centroid
            y_exp = func(x_exp)
            # if the the further point is better, accept it.
            if y_exp < y_new:
                x_points[-1] = x_exp
                y_vals[-1] = y_exp
            # else take the initial new point.
            else:
                x_points[-1] = x_new
                y_vals[-1] = y_new

        # if the new point would still be the worst:
        else:
            # get closer to the centroid, instead of mirroring
            x_new = 0.5 * (centroid + x_points[-1])
            y_new = func(x_new)
            # if this propsed point is better, overwrite it
            if y_new < y_vals[-1]:
                x_points[-1] = x_new
                y_vals[-1] = y_new
            # if this point is still not better:
            else:
                # shrink all points around the best point so far
                x_points[1:] = 0.5 * (x_points[0] + x_points[1:])
                # update all y_vals
                for i in range(1, N + 1):
                    y_vals[i] = func(x_points[i])

    print("WARNING: Desired tolerance for downhill-simplex not reached")

    return x_points[0]
