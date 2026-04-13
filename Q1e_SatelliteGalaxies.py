# imports
import numpy as np
import matplotlib.pyplot as plt

from chi2 import chi2
from romberg_integrator import romberg_integrator
from distribution import Distribution
from downhill_simplex import downhill_simplex
from minimizer import Minimizer


def readfile(filename):
    """
    Helper function to read in the satellite galaxy data from the provided text files.

    Parameters
    ----------
    filename : str
        The name of the file to read in.

    Returns
    -------
    radius : ndarray
        The virial radius for all the satellites in the file.
    nhalo : int
        The number of halos in the file.
    """
    f = open(filename, "r")
    data = f.readlines()[3:]  # Skip first 3 lines
    nhalo = int(data[0])  # number of halos
    radius = []

    for line in data[1:]:
        if line[:-1] != "#":
            radius.append(float(line.split()[0]))

    radius = np.array(radius, dtype=float)
    f.close()
    return (
        radius,
        nhalo,
    )  # Return the virial radius for all the satellites in the file, and the number of halos


def n(x: np.ndarray, A: float, Nsat: float, a: float, b: float, c: float) -> np.ndarray:
    """
    Number density profile of satellite galaxies

    Parameters
    ----------
    x : ndarray
        Radius in units of virial radius; x = r / r_virial
    A : float
        Normalisation
    Nsat : float
        Average number of satellites
    a : float
        Small-scale slope
    b : float
        Transition scale
    c : float
        Steepness of exponential drop-off

    Returns
    -------
    ndarray
        Same type and shape as x. Number density of satellite galaxies
        at given radius x.
    """
    b_inv = 1 / b
    return A * Nsat * (x * b_inv) ** (a - 3) * np.exp(-((x * b_inv) ** c))


def satellite_number(
    x: np.ndarray, A: float, Nsat: float, a: float, b: float, c: float
) -> np.ndarray:
    """
    Expected number of satellite galaxies at radius x.

    Parameters
    ----------
    x : ndarray
        Radius in units of virial radius; x = r / r_virial
    A : float
        Normalisation
    Nsat : float
        Average number of satellites
    a : float
        Small-scale slope
    b : float
        Transition scale
    c : float
        Steepness of exponential drop-off

    Returns
    -------
    ndarray
        Same type and shape as x. Expected number of satellite galaxies at radius x.
    """
    return 4 * np.pi * x * x * n(x, A, Nsat, a, b, c)


#### Fitting ####


def get_normalization_constant(a: float, b: float, c: float, order: int = 5) -> float:
    """
    Calculate the normalization constant A (which is a function of a,b,c) for the satellite number density profile.

    Parameters
    ----------
    a : float
        Small-scale slope.
    b : float
        Transition scale.
    c : float
        Steepness of exponential drop-off.
    Nsat : (float, optional)
        Average number of satellites. Defaults to 1.
    order: int
        Order to use for romberg integrator.

    Returns
    -------
    float
        Normalization constant A.
    """
    # integrate in logspace
    func = lambda x: satellite_number(x=x, A=1, Nsat=1, a=a, b=b, c=c)
    I = romberg_integrator(func, (0, 5), order=order)
    return 1 / I


def g(
    a: float, b: float, c: float, data: np.ndarray, normalization_order: int = 5
) -> float:
    """
    returns the g_function value (See overleaf document).
    A minimum of this function
    corresponds to the optimal parametrization a, b, c
    such that it minimizes the poisson likelihood.

    Args:
        a (float):
        b (float):
        c (float):
        data (np.ndarray):
            array containing data with the satellite radii
        normalization_order (int, optional):
            Order to use for normalizing the models. Defaults to 5.

    Returns:
        g_value (float):
    """
    # normalize the model:
    # we do this by setting N_sat to 1.
    A = get_normalization_constant(a, b, c, order=normalization_order)
    # Sum up the log terms as described within the overleaf document
    ln_terms = np.log(satellite_number(data, A, 1, a, b, c))
    return -np.sum(ln_terms)


# =====================================================
# ======== Main functions for each subquestion ========
# =====================================================


def do_question_1e():
    # ======== Question 1e: Monte Carlo simulations ========
    # pick one of the data files to perform the Monte Carlo simulations on, e.g. m12
    datafiles = ["m11", "m12", "m13", "m14", "m15"]
    index = (
        3 # index of the data file to use for Monte Carlo simulations, e.g. 1 for m12
    )

    radius, nhalo = readfile(f"Data/satgals_{datafiles[index]}.txt")
    Nsat = len(radius) / nhalo

    # read in the best parameters from question b
    best_params_chi2 = []
    with open("Calculations/best_params_chi2.txt", "r") as f:
        for line in f:
            a, b, c = map(float, line.split())
            best_params_chi2.append([a, b, c])

    # read in the best parameters from question c
    best_params_poisson = []
    with open("Calculations/best_params_poisson.txt", "r") as f:
        for line in f:
            a, b, c = map(float, line.split())
            best_params_poisson.append([a, b, c])

    # we now bin the radii found in the data
    # we choose our bin ranges slightly larger
    # than how far our data extends, because empty bins
    # contain information.
    nbins = 10
    x_lower, x_upper = (
        np.min(radius),
        np.max(radius),
    )
    bin_edges = np.geomspace(x_lower, x_upper, nbins + 1)
    bin_widths = bin_edges[1:] - bin_edges[:-1]

    # Use best-fit parameters from previous steps for the original data file
    best_params_chi2 = best_params_chi2[index]
    best_params_poisson = best_params_poisson[index]

    # compute normalization constants corresponding to best-fit parameters
    A_chi2 = get_normalization_constant(*best_params_chi2, order=10)
    A_poisson = get_normalization_constant(*best_params_poisson, order=10)

    N_chi2 = lambda x: satellite_number(x, A_chi2, 1, *best_params_chi2)
    N_poisson = lambda x: satellite_number(x, A_poisson, 1, *best_params_poisson)

    # initialize as distribution object, which will give us the samples
    chi2_sampler = Distribution(N_chi2, xmin=x_lower, xmax=x_upper, seed=1)
    poisson_sampler = Distribution(N_poisson, xmin=x_lower, xmax=x_upper, seed=2)

    # we can also compute the max values that these distributions obtain
    # using the methodology from exercise a, we will use these for
    # slightly more efficient rejection sampling:

    # instantiate minimizer object
    # first for chi2 parameters:
    # do minimization in log space for speed
    minimizer_chi2 = Minimizer(func=lambda logx: -N_chi2(np.exp(logx)))
    bracket_chi2 = minimizer_chi2.bracket(
        -5, 2
    )  # find bracket, starting from two guessed x_values

    logx_max_chi2 = minimizer_chi2.tighten(
        bracket_chi2
    )  # find the log value of the x at which maximum of N(x) is attained
    x_max_chi2 = np.exp(logx_max_chi2)  # convert to x_value
    N_max_chi2 = satellite_number(x_max_chi2, A_chi2, 1, *best_params_chi2)

    # now for poisson parameters:
    # do minimization in log space for speed
    minimizer_poisson = Minimizer(func=lambda logx: -N_poisson(np.exp(logx)))
    bracket_poisson = minimizer_poisson.bracket(
        -5, 2
    )  # find bracket, starting from two guessed x_values

    logx_max_poisson = minimizer_poisson.tighten(
        bracket_poisson
    )  # find the log value of the x at which maximum of N(x) is attained
    x_max_poisson = np.exp(logx_max_poisson)  # convert to x_value
    N_max_poisson = satellite_number(x_max_poisson, A_poisson, 1, *best_params_poisson)

    # initialize lists in which we will store the best fit parameters to the
    # generated pseudo-data
    pseudo_chi2_params = []
    pseudo_poisson_params = []

    num_pseudo_experiments = 30 # replace by number with reasonable runtime
    for i in range(num_pseudo_experiments):

        # sample a pseudo_data set from the two samplers.
        # set pmax ever so slightly larger than computed, to account for error
        pseudo_radius_chi2 = chi2_sampler.rejection(
            N_samples=len(radius), pmax=1.01 * N_max_chi2
        )
        pseudo_radius_poisson = poisson_sampler.rejection(
            N_samples=len(radius), pmax=1.01 * N_max_poisson
        )

        # construct the y_data for chi2:
        # mean number of satellites per halo in each radial bin, Ni
        bin_counts_chi2 = np.histogram(pseudo_radius_chi2, bin_edges)[
            0
        ]  # "mean number of satellites in each radial bin"
        # we divide by bin_widths so the data height does not depend on choice of bins
        Ntilde_data_chi2 = bin_counts_chi2 / (nhalo * bin_widths)  # "per halo"

        ##
        # chi2 fitting:
        ##

        # instantiate the initial guesses for the parametrizations
        # note that this combination of initial guess is NOT degenerate
        a = 2.4
        b = 0.4
        c = 1.6
        x_init = np.array(
            [(a, b, c), (1.5 * a, b, c), (a, 1.5 * b, c), (a, b, 1.5 * c)]
        )

        def chi2_temp(abc):
            # always recompute the normalization constant when trying new parameters
            A_temp = get_normalization_constant(*abc, order=10)

            def model(bin):
                func = lambda x: satellite_number(x, A_temp, Nsat, *abc)
                # integrate the number counts over the bin, set order low for computational speed.
                bounds = (bin_edges[bin], bin_edges[bin + 1])
                I = romberg_integrator(func, bounds=bounds, order=7)
                return I / bin_widths[bin]

            Ntilde_model = np.array([model(bin) for bin in np.arange(nbins)])
            err_data = np.sqrt(Ntilde_model)  # poissonian property

            return np.sum(
                (Ntilde_data_chi2 - Ntilde_model)
                * (Ntilde_data_chi2 - Ntilde_model)
                / (err_data * err_data)
            )

        # use downhill simplex to find parametrization that
        # minimizes chi2
        p_opt_chi2 = downhill_simplex(chi2_temp, x_init=x_init, relerr=1e-4)
        pseudo_chi2_params.append(p_opt_chi2)

        ##
        # poisson fitting:
        ##

        # use downhill simplex to find parametrization that
        # minimizes g
        p_opt_poisson = downhill_simplex(
            lambda abc: g(*abc, pseudo_radius_poisson), x_init=x_init
        )
        pseudo_poisson_params.append(p_opt_poisson)

    # plot the pseudo best-fit profiles, plot the original best-fit profile in another color and plot the mean in one more color

    # chi2 plot
    x_plot = np.linspace(np.min(pseudo_radius_chi2), np.max(pseudo_radius_chi2), 100)  # create x_array for plotting the model
    plt.figure(figsize=(6.4, 4.8))
    for params in pseudo_chi2_params:
        plt.plot(
            x_plot,
            satellite_number(
                x_plot, get_normalization_constant(*params), Nsat, *params
            ),
        )  # plot the best-fit model for each pseudo-dataset using the best-fit parameters found from chi-squared minimization

    plt.plot(
        x_plot,
        satellite_number(
            x_plot,
            get_normalization_constant(*best_params_chi2),
            Nsat,
            *best_params_chi2,
        ),
    )  # plot the original best-fit model using the best-fit parameters found from chi-squared minimization on the real data

    mean_params_chi2 = np.mean(
        pseudo_chi2_params, axis=0
    )  # calculate the mean of the best-fit parameters from the pseudo-datasets
    plt.plot(
        x_plot,
        satellite_number(
            x_plot,
            get_normalization_constant(*mean_params_chi2),
            Nsat,
            *mean_params_chi2,
        ),
    )  # plot the mean of the best-fit models from the pseudo-datasets

    plt.title(f"Monte Carlo simulations - chi2 fit - Data file: {datafiles[index]}")
    plt.xlabel("x = r / r_virial")
    plt.ylabel("Number of satellites")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(["Pseudo-dataset fits", "Original fit", "Mean of pseudo fits"])
    plt.savefig("Plots/satellite_monte_carlo_chi2.png")

    # poisson plot
    x_plot = np.linspace(np.min(pseudo_radius_poisson), np.max(pseudo_radius_poisson), 100)  # create x_array for plotting the model
    plt.figure(figsize=(6.4, 4.8))
    for params in pseudo_poisson_params:
        plt.plot(
            x_plot,
            satellite_number(
                x_plot, get_normalization_constant(*params), Nsat, *params
            ),
        )  # plot the best-fit model for each pseudo-dataset using the best-fit parameters found from Poisson negative log-likelihood minimization
    plt.plot(
        x_plot,
        satellite_number(
            x_plot,
            get_normalization_constant(*best_params_poisson),
            Nsat,
            *best_params_poisson,
        ),
    )  # plot the original best-fit model using the best-fit parameters found from Poisson negative log-likelihood minimization on the real data

    mean_params_poisson = np.mean(
        pseudo_poisson_params, axis=0
    )  # calculate the mean of the best-fit parameters from the pseudo-datasets
    plt.plot(
        x_plot,
        satellite_number(
            x_plot,
            get_normalization_constant(*mean_params_poisson),
            Nsat,
            *mean_params_poisson,
        ),
    )  # plot the mean of the best-fit models from the pseudo-datasets

    plt.title(f"Monte Carlo simulations - Poisson fit - Data file: {datafiles[index]}")
    plt.xlabel("x = r / r_virial")
    plt.ylabel("Number of satellites")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(["Pseudo-dataset fits", "Original fit", "Mean of pseudo fits"])
    plt.savefig("Plots/satellite_monte_carlo_poisson.png")


if __name__ == "__main__":
    # do_question_1a()
    # do_question_1b()
    # do_question_1c()
    # do_question_1d()
    do_question_1e()
