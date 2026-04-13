# imports
import numpy as np
import matplotlib.pyplot as plt

from chi2 import chi2
from romberg_integrator import romberg_integrator
from gradient import gradient
from matrix import Matrix


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


def levenberg_marquardt_satellites(
    y_data: np.ndarray,
    bin_edges: np.ndarray,
    Nsat: float,
    p_init: np.ndarray,
    lmbda_init: float = 1e-3,
    w=10,
    maxit: int = 1000,
    rel_tol: float = 1e-4,
    abs_tol: float = 1e-4,
) -> np.ndarray:
    """
    levenberg-marquardt algorithm to find parameters that minimize
    chi^2 for the given data and model, adjusted to find a, b, c for satellites.

    Args:
        model (callable):
            model of which to find optimal parameters
        y_data (np.ndarray):
            counts in each bin.
        bins (np.ndarray):
            x_values of bin_edges.
        Nsat (float):
            average number of satellites per halo
        p_init (np.ndarray):
            initial estimate of optimal parameters.
        lmbda_init (float, optional):
            initial lambda_value, sets the step value for updates. Defaults to 1e-3.
        w (int, optional):
            sets how quickly lmbda updates. Defaults to 10.
        maxit (int, optional):
            maximum iterations the algorithm may take. Defaults to 1000.
        reltol (float, optional):
            relative tolerance (if chi_new - chi < reltol*chi algorithm terminates).
            Defaults to 1e-8.
        abstol (float, optional):
            absolute tolerance  (if chi_new - chi < abstol algorithm terminates).
            Defaults to 1e-4.

    Returns:
        p (np.ndarray):
            array containing optimal parameters.
        min_Chi2 (int):
            minimum value of Chi2 found.
    """

    # create the bins array, which simply labels each bin
    bins = np.array(
        range(0, len(bin_edges) - 1)
    )  # note this has len(bin_edges)-1 elements
    bin_widths = bin_edges[1:] - bin_edges[:-1]

    lmbda = lmbda_init
    p = p_init.copy()

    # the model expected value in bin i is given by \int_(x_i)^(x_i+1) N(x) dx
    # where x_i, x_i+1 are the lower and upper bounds of bin i.
    
    def model(bin, a, b, c):
        # always recompute the normalization constant when trying new parameters
        A_temp = get_normalization_constant(a, b, c)
        
        func = lambda x: satellite_number(x=x, A=A_temp, Nsat=Nsat, a=a, b=b, c=c)
        
        # integrate the number counts over the bin, set order low for computational speed.
        bounds = (bin_edges[bin], bin_edges[bin + 1])
        I = romberg_integrator(func, bounds=bounds, order=4)
        
        return I / bin_widths[bin]

    err_data = np.sqrt(np.array([model(bin, *p) for bin in np.arange(len(y_data))]))

    # compute chi
    chi = chi2(model, y_data, bins, err_data, p)
    # precompute 1/w since we will be using this multiple times:
    w_inv = 1 / w

    for j in range(maxit):
        # create chi function that only depends on model parameters:
        chi_temp = lambda args: chi2(model, y_data, bins, err_data, args)
            
        # compute the gradient of chi_temp at p and compute beta
        beta = -0.5 * gradient(chi_temp, p, h=1e-2)

        # initialize alpha matrix of size (params, params,):
        alpha = np.zeros(
            (
                len(p),
                len(p),
            ),
            dtype=float,
        )
        # loop over data points:
        for i in range(len(bins)):
            # create model function that only depends on model parameters:
            model_temp = lambda args: model(bins[i], *args)
            # compute the gradient of model_temp at p
            model_gradient = gradient(model_temp, p, h=1e-2)
            # error is given by model expected value: poissonian
            alpha += np.outer(model_gradient, model_gradient) / (
                err_data[i] * err_data[i]
            )

        # add lmbda*(diagonal of alpha), this is what makes it the levenberg algorithm
        # np.diag(np.diag()) extracts the diagonal of a matrix
        alpha += lmbda * np.diag(np.diag(alpha))

        # make alpha a matrix object to solve alpha@delp = beta:
        alpha = Matrix(alpha)
        delp = alpha.solve(beta)

        # propose a new parametrization of the model
        p_new = p + delp

        # redefine the model with the new normalization constant.
        def model_new(bin, a, b, c):
            
            # always recompute normalization constant
            A_temp = get_normalization_constant(a,b,c, order=7)
            
            func = lambda x: satellite_number(x=x, A=A_temp, Nsat=Nsat, a=a, b=b, c=c)
            
            # integrate the number counts over the bin, set order for computational speed.
            I = romberg_integrator(func, (bin_edges[bin], bin_edges[bin + 1]), order=4)
            
            return I / bin_widths[bin]

        err_data_new = np.sqrt(
            np.array([model_new(bin, *p_new) for bin in np.arange(len(y_data))])
        )
        chi_new = chi2(model_new, y_data, bins, err_data_new, p_new)

        # check termination condition:
        Delta_chi = chi_new - chi
        if (
            np.abs(Delta_chi) < rel_tol * chi or np.abs(Delta_chi) < abs_tol
        ):  # chi - chi_new > 0
            return p + delp, chi_new

        # check whether new parametrization is better
        if Delta_chi < 0:  # if new point is better:
            # otherwise update step
            p += delp
            chi = chi_new
            model = model_new
            err_data_new = err_data
            lmbda *= w_inv  # become more like newton's method

        else:  # if new point is worse:
            lmbda *= w  # become more like steepest descent

    print("WARNING (levenberg-marquardt): desired tolerance not reached")
    return p, chi


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
        nbins = 20
        x_lower, x_upper = (
            np.min(radius) * 0.5,
            np.max(radius) * 2,
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

        # set the initial guess for abc, these starting values seemed to converge well.
        a = 1.5
        b = 0.8
        c = 2.5
        p_init = np.array([a, b, c])

        # now use levenberg_marquardt to find the optimal fit,
        p_opt, min_chi2 = levenberg_marquardt_satellites(
            y_data=Ntilde_data,
            bin_edges=bin_edges,
            Nsat=Nsat,
            p_init=p_init,
            lmbda_init=1e-3,
            w=10,
            maxit=1000,
            rel_tol=1e-7,
            abs_tol=1e-5,
        )

        min_chi2_values.append(min_chi2)
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
        ax.set_ylim(1e-4, None)
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
