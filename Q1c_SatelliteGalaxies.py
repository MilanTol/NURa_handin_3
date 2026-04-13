# imports
import numpy as np
import matplotlib.pyplot as plt

from romberg_integrator import romberg_integrator
from downhill_simplex import downhill_simplex


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


def log_factorial(x: int):
    result = 0.0
    for i in range(2, x + 1):
        result += np.log(i)
    return result


def negative_poisson_ln_likelihood(
    model: callable, x_data: np.ndarray, y_data: np.ndarray, params: tuple
) -> float:
    """
    Calculate the Poisson negative log-likelihood for a given set of parameters and data.

    Parameters
    ----------
    model : callable
        The model function to compare to the data.
    data : ndarray
        The observed data to compare the model to.
    params : tuple
        The parameters to evaluate the model at.

    Returns
    -------
    float
        The Poisson negative log-likelihood value for the given parameters and data.
    """

    # use map to make a run log_factorial vectorized
    log_factorial_terms = np.array([log_factorial(y) for y in y_data])
    terms = (
        y_data * np.log(model(x_data, *params))
        - model(x_data, *params)
        - log_factorial_terms
    )
    return -np.sum(terms)


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


def do_question_1c():
    # ======== Question 1c: Fitting N(x) with Poisson ln-likelihood ========
    datafiles = ["m11", "m12", "m13", "m14", "m15"]

    min_poisson_llh_values = []
    best_params_poisson = []

    # initialize figure with 5 subplots on 3x2 grid for the 5 data files
    fig, axs = plt.subplots(3, 2, figsize=(6.4, 8.0))
    axs = axs.flatten()

    for datafile in datafiles:
        radius, nhalo = readfile(f"Data/satgals_{datafile}.txt")

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

        # the x_value at which we got the data should be the center of the bin,
        # however we have logspace bins. So we instead choose the center in logspace:
        logspace_centers = 0.5 * (np.log10(bin_edges[:-1]) + np.log10(bin_edges[1:]))
        # then we get the x_values by exponentiating
        bin_centers = 10**logspace_centers

        # construct the y_data as:
        # mean number of satellites per halo in each radial bin, Ni
        bin_counts = np.histogram(radius, bin_edges)[
            0
        ]  # "mean number of satellites in each radial bin"
        Ntilde_data = bin_counts / (nhalo * bin_widths)  # "per halo"

        # we compute the average number of satellites per halo
        # by computing the length of the radius array (which is equal to the number of
        # satellites) and then dividing by the total number of halos (nhalo).
        Nsat = np.sum(bin_counts) / nhalo

        # instantiate the initial guesses for the parametrizations
        # note that this combination of initial guess is NOT degenerate
        a = 2.4
        b = 0.4
        c = 1.6
        x_init = np.array(
            [(a, b, c), (1.5 * a, b, c), (a, 1.5 * b, c), (a, b, 1.5 * c)]
        )

        # we want to minimize g, using the current data set
        func = lambda abc: g(*abc, radius, normalization_order=10)
        # find the optimal parameters with downhill_simplex
        p_opt = downhill_simplex(func, x_init=x_init)

        # compute the negative log likelihood:
        Nsat = len(radius) / nhalo
        model = lambda x: satellite_number(
            x, get_normalization_constant(*p_opt), Nsat, *p_opt
        )
        neg_llh = negative_poisson_ln_likelihood(
            model, bin_centers, np.int64(Ntilde_data), params=()
        )

        # Store poisson llh values and best-fit parameters in their arrays
        min_poisson_llh_values.append(neg_llh)
        best_params_poisson.append(p_opt)

        # instantiate plot
        ax = axs[datafiles.index(datafile)]
        x_plot = np.linspace(  # create x_array for plotting the model
            x_lower, x_upper, 100
        )
        ax.stairs(Ntilde_data, edges=bin_edges, label="data")
        ax.plot(  # plot the model that maximizes Poisson likelihood
            x_plot,
            satellite_number(x_plot, get_normalization_constant(*p_opt), Nsat, *p_opt),
            label="model",
        )

        # Add labels and title to the subplot
        ax.set_title(f"Data file: {datafile}")
        ax.set_xlabel("x = r / r_virial")
        ax.set_ylabel("Number of satellites")

        # log-log scaling
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.5*np.min(Ntilde_data), 2*np.max(Ntilde_data))
        ax.legend()

    # Save the figure with all subplots
    plt.tight_layout()
    plt.savefig("Plots/satellite_fits_poisson.png")

    # Save poisson llh values and best-fit parameters for each data file to text files for later use in the PDF
    with open("Calculations/table_fitparams_poisson.tex", "w") as f:
        rows = list(zip(min_poisson_llh_values, best_params_poisson))
        for idx, (llh_val, params) in enumerate(rows):
            a, b, c = params
            line_end = " \\\\" if idx < len(rows) - 1 else ""
            f.write(
                f"m{idx+11} & {llh_val:.5f} & {a:.5f} & {b:.5f} & {c:.5f}{line_end}\n"
            )

    # store the best fit parameters in a format that we can read for question d
    with open("Calculations/best_params_poisson.txt", "w") as f:
        for a, b, c in best_params_poisson:
            f.write(f"{a} {b} {c}\n")


if __name__ == "__main__":
    do_question_1c()
