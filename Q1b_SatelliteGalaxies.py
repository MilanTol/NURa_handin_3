# imports
import numpy as np
import matplotlib.pyplot as plt

from chi2 import chi2
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


def get_normalization_constant(a: float, b: float, c: float, order: int = 6) -> float:
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


def do_question_1b():
    # ======== Question 1b: Fitting N(x) with chi-squared ========
    datafiles = ["m11", "m12", "m13", "m14", "m15"]  # ["m13", "m14", "m15"]

    N_sat = []
    best_params_chi2 = []
    min_chi2_values = []

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

        # construct the y_data as:
        # mean number of satellites per halo in each radial bin, Ni
        bin_counts = np.histogram(radius, bin_edges)[
            0
        ]  # "mean number of satellites in each radial bin"
        # we divide by bin_widths so the data height does not depend on choice of bins
        Ntilde_data = bin_counts / (nhalo * bin_widths)  # "per halo"

        # we compute the average number of satellites per halo
        # by computing the length of the radius array (which is equal to the number of
        # satellites) and then dividing by the total number of halos (nhalo).
        Nsat = len(radius) / nhalo
        N_sat.append(Nsat)

        # to compute the error on the data, we use that number counts in bins is a
        # Poissonian process, hence the error is given by 1/sqrt(y_data)
        # however we can't have 0 error, so we take a minimum error of 1:

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
                (Ntilde_data - Ntilde_model)
                * (Ntilde_data - Ntilde_model)
                / (err_data * err_data)
            )

        # use downhill simplex to find parametrization that
        # minimizes chi2
        p_opt = downhill_simplex(chi2_temp, x_init=x_init)

        # store the best smallest chi2 value found and the
        # corresponding parametrization
        min_chi2_values.append(chi2_temp(p_opt))
        best_params_chi2.append(p_opt)

        ax = axs[datafiles.index(datafile)]
        # plot the data
        ax.stairs(Ntilde_data, edges=bin_edges, label="data")
        x_plot = np.geomspace(x_lower, x_upper, 100)
        # plot the best-fit model using the best-fit parameters found from chi-squared minimization
        ax.plot(
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
        ax.set_ylim(0.5 * np.min(Ntilde_data), 2 * np.max(Ntilde_data))
        ax.legend()

    # Save the figure with all subplots
    plt.tight_layout()
    plt.savefig("Plots/satellite_fits_chi2.png")

    # Save N_sat, chi2 values and best-fit parameters for each data file to tex files for later use in the PDF
    with open("Calculations/table_fitparams_chi2.tex", "w") as f:
        rows = list(zip(N_sat, min_chi2_values, best_params_chi2))
        for idx, (N_sat_temp, chi2_val, params) in enumerate(rows):
            a, b, c = params
            line_end = " \\\\" if idx < len(rows) - 1 else ""
            f.write(
                f"m{idx+11} & {N_sat_temp:.5f} & {chi2_val:.5f} & {a:.5f} & {b:.5f} & {c:.5f}{line_end}\n"
            )

    # store the best fit parameters in a format that we can read for question d
    with open("Calculations/best_params_chi2.txt", "w") as f:
        for a, b, c in best_params_chi2:
            f.write(f"{a} {b} {c}\n")


if __name__ == "__main__":
    do_question_1b()
