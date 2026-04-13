import numpy as np
from rng import RNG


class Distribution:
    def __init__(
        self, dist: callable, xmin: float, xmax: float, args: tuple = (), seed=1
    ):
        """
        Parameters
        dist : callable
            function that is proportional to distribution, must not be negative anywhere.
        min :
            Minimum value for sampling
        max : float
            Maximum value for sampling
        args : tuple, optional
            Arguments of the distribution to sample, passed as args to dist
        seed : int, optional
            Seed used for random number generation

        """
        self.dist = dist
        self.xmin = xmin
        self.xmax = xmax
        self.args = args
        self.rng = RNG(seed)

    def __call__(self, x):
        return self.dist(x, *self.args)

    def rejection(
        self, N_samples: int = 1, pmax: float = 1, rej_samples: bool = False
    ) -> np.ndarray:
        """
        samples x values from the distribution using rejection sampling.
        The distribution must not exceed p_max.

        Parameters
        Nsamples : int
            Number of samples
        p_max : float
            upper bound for probability distribution
        rej_samples: bool
            whether to also return the number of rejected samples

        Returns
        -------
        sample: ndarray
            Values sampled from dist, shape (Nsamples,)
        """
        samples = []
        samples_collected = 0
        samples_rejected = 0
        while samples_collected < N_samples:
            x = self.rng.float(
                (self.xmin, self.xmax)
            )  # randomly draw an x in between bounds
            y = self.rng.float((0, pmax))  # draw a y in between 0 and pmax
            dist_val = self.dist(x, *self.args)
            if y < dist_val:
                samples.append(x)
                samples_collected += 1
            else:
                samples_rejected += 1

        if rej_samples:
            return np.array(samples), samples_rejected
        return np.array(samples)
