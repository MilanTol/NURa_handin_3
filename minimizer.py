import numpy as np


def parabola_minimum(a, b, c, f_a, f_b, f_c):
    """
    returns x at which quadratic fit to 3 data points has a minimum.
    """
    num = 0.5 * (b - a) ** 2 * (f_b - f_c) - (b - c) ** 2 * (f_b - f_a)
    den = (b - a) * (f_b - f_c) - (b - c) * (f_b - f_a)
    return b - num / den


class Minimizer:
    def __init__(self, func: callable, args: tuple = ()):
        self.func = func
        self.args = args

    def bracket(
        self, a: float, b: float, maxit: int = 100
    ) -> tuple[float, float, float]:
        """
        returns a bracket (a,b,c) containing a minimum.

        Args:
            a (float): initial guess for lower x_value
            b (float): initial guess for upper x_value
            maxit (int, optional): max iterations to find a bracket. Defaults to 100.

        Raises:
            Exception: If it does not find a bracket within max iterations, it raises an exception.

        Returns:
            tuple[float, float, float]: x_values that form a bracket
        """

        f_a = self.func(a, *self.args)
        f_b = self.func(b, *self.args)

        if f_a < f_b:  # ensure that f(b) < f(a). If not, then swap a and b around.
            a, b = b, a
            f_a, f_b = f_b, f_a

        phi = 1.618  # use golden ratio as fraction of step size
        c = b + (b - a) * phi  # propose a new point c
        f_c = self.func(c, *self.args)

        if f_c > f_b:  # check whether a, b, c forms a bracket. If so, return abc
            return a, b, c

        for i in range(maxit):

            # If abc do not form a bracket:
            d = parabola_minimum(
                a, b, c, f_a, f_b, f_c
            )  # fit parabola to abc, store minimum
            f_d = self.func(d, *self.args)

            if b < d and d < c:  # check whether d is in between b and c.
                if f_d < f_c:  # check whether bdc forms a bracket
                    return b, d, c
                if f_d > f_b:  # check whether abd forms a bracket
                    return a, b, d
                else:  # -> f_d is in between f_b and f_c
                    d = c + (c - b) * phi
            else:  # -> d lies beyond c
                if np.abs(d - b) > 100 * np.abs(
                    c - b
                ):  # check whether its too far beyond.
                    d = c + (c - b) * phi  # if so, reject d and sample a new d.

            a, b, c = b, c, d  # readjust all points
            f_a, f_b, f_c = f_b, f_c, f_d

            if f_c > f_b:  # check whether a, b, c forms a bracket. If so, return abc
                return a, b, c

        raise Exception(f"could not find bracket in {maxit} iterations")

    def tighten(
        self,
        bracket: tuple[float, float, float],
        abserr: float = 1e-5,
        relerr: float = 1e-5,
        maxit: int = 100,
    ) -> float:
        """
        Returns the x_val at which the minimum resides inside a bracket by golden section search algorithm.

        Args:
            bracket (tuple[float,float,float]): bracket (must contain a, b, c)
            abserr (float, optional): _description_. Defaults to 1e-5
            relerr (float, optional): _description_. Defaults to 1e-5
            maxit (int, optional): _description_. Defaults to 100.

        Raises:
            Exception: If it does not find a bracket within max iterations, it raises an exception.

        Returns:
            float: returns x_value at which minimum is located.
        """
        a, b, c = bracket  # unpack bracket tuple
        f_b = self.func(b, *self.args)

        if np.abs(c - b) > np.abs(b - a):  # identify larger half of the bracket
            x = c  # use interval (b, c)
        else:
            x = a  # use interval (a, b)

        w = 0.38197  # = 1/(1+phi), where phi is golden ratio

        for i in range(maxit):
            d = b + (x - b) * w  # propose a new point d within the selected interval
            f_d = self.func(d, *self.args)

            if (
                np.abs(c - a) < abserr or np.abs((c - a) * b) < relerr
            ):  # check whether target accuracy is reached
                # return d or b depending on which is minimum
                if f_d < f_b:
                    return d
                else:
                    return b

            # if target accuracy is not reached:
            if f_d < f_b:  # tighten towards d:
                if x == a:
                    c = b
                    b, f_b = d, f_d  # Note that x remains a
                else:
                    a = b
                    b, f_b = d, f_d  # Note that x remains c
            else:  # tighten towards b:
                if x == a:
                    a = d
                    x = c  # change x to c since that is now the largest gap
                else:
                    c = d
                    x = a  # change x to a since that is now the largest gap

        # perform one final iteration with new points:
        d = b + (x - b) * w  # propose a new point d within the selected interval
        f_d = self.func(d, *self.args)

        if (
            np.abs(c - a) < abserr or np.abs(c - a) * b < relerr
        ):  # check whether target accuracy is reached
            # return d or b depending on which is minimum
            if f_d < f_b:
                return d
            else:
                return b

        raise Exception(
            f"could not find minimum with expected tolerance in {maxit+1} iterations"
        )
