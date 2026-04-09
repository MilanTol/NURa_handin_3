# imports
import numpy as np
import matplotlib.pyplot as plt

from chi2 import chi2
from minimizer import Minimizer
from romberg_integrator import romberg_integrator
from downhill_simplex import downhill_simplex
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


def levenberg_marquardt_satellites(
    y_data: np.ndarray,
    bin_edges: np.ndarray,
    Nsat: int,
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
            y_values of data.
        x_data (np.ndarray):
            x_values of data.
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

    # compute the normalization constant for initial parametrization
    A = get_normalization_constant(*p, order=7)

    # the model expected value in bin i is given by \int_(x_i)^(x_i+1) N(x) dx
    # where x_i, x_i+1 are the lower and upper bounds of bin i.
    def model(bin, a, b, c):
        func = lambda x: satellite_number(x=x, A=A, Nsat=Nsat, a=a, b=b, c=c)
        # integrate the number counts over the bin, set order low for computational speed.
        bounds = (bin_edges[bin], bin_edges[bin + 1])
        I = romberg_integrator(func, bounds=bounds, order=3)
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
        A_new = get_normalization_constant(*p_new, order=7)

        # redefine the model with the new normalization constant.
        def model_new(bin, a, b, c):
            func = lambda x: satellite_number(x=x, A=A_new, Nsat=Nsat, a=a, b=b, c=c)
            # integrate the number counts over the bin, set order for computational speed.
            I = romberg_integrator(func, (bin_edges[bin], bin_edges[bin + 1]), order=5)
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
    return p, chi_new


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


def Gtest(y_data: np.ndarray, y_model: np.ndarray) -> float:
    """
    returns Gtest estimate of data and model

    Args:
        y_data (np.ndarray): _array containing data
        y_model (np.ndarray): array containing model expected values

    Returns:
        float: Gtest
    """
    return 2 * np.sum(y_data * np.log(y_data / y_model))


# =====================================================
# ======== Main functions for each subquestion ========
# =====================================================


def do_question_1a():
    # ======== Question 1a: Maximization of N(x) ========
    a = 2.4
    b = 0.25
    c = 1.6
    Nsat = 100
    A_1a = 256 / (5 * np.pi ** (3 / 2))
    # x_lower, x_upper = 10**-4, 5

    # set N in logspace to make computations faster, use -N since were finding maximum with minimizer.
    negative_N_logspace = lambda x: -satellite_number(np.exp(x), A_1a, Nsat, a, b, c)

    # Plot N, to inspect the function
    x_logvals = np.geomspace(1e-4, 5, 100)
    plt.plot(x_logvals, -negative_N_logspace(np.log(x_logvals)))
    plt.ylabel("N(x)")
    plt.xlabel("x")
    plt.yscale("log")
    plt.xscale("log")
    plt.ylim(1e-6, None)
    plt.savefig("Plots/1a_N(x)")
    plt.close()

    # instantiate minimizer object
    minimizer = Minimizer(func=negative_N_logspace)
    bracket = minimizer.bracket(
        -4, 1
    )  # find bracket, starting from two guessed x_values

    logx_max = minimizer.tighten(
        bracket
    )  # find the log value of the x at which maximum of N(x) is attained
    x_max = np.exp(logx_max)  # convert to x_value
    Nx_max = satellite_number(x_max, A_1a, Nsat, a, b, c)

    # Write the results to text files for later use in the PDF
    with open("Calculations/satellite_max_x.txt", "w") as f:
        f.write(f"{x_max:.6f}")
    with open("Calculations/satellite_max_Nx.txt", "w") as f:
        f.write(f"{Nx_max:.6f}")


best_params_chi2 = []


def do_question_1b():
    # ======== Question 1b: Fitting N(x) with chi-squared ========
    datafiles = ["m11", "m12", "m13", "m14", "m15"]  # ["m13", "m14", "m15"]

    N_sat = []
    min_chi2_values = []

    # initialize figure with 5 subplots on 3x2 grid for the 5 data files
    fig, axs = plt.subplots(3, 2, figsize=(6.4, 8.0))
    axs = axs.flatten()

    for datafile in datafiles:
        radius, nhalo = readfile(f"Data/satgals_{datafile}.txt")

        print("computing:", datafile)

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
        Ntilde_data = bin_counts / (nhalo * bin_widths)  # "per halo"

        # we compute the average number of satellites per halo
        # by computing the length of the radius array (which is equal to the number of
        # satellites) and then dividing by the total number of halos (nhalo).
        Nsat = np.sum(bin_counts) / nhalo
        N_sat.append(Nsat)

        # to compute the error on the data, we use that number counts in bins is a
        # Poissonian process, hence the error is given by 1/sqrt(y_data)
        # however we can't have 0 error, so we take a minimum error of 1:

        # set the initial guess for abc
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
            lmbda_init=1e-5,
            w=10,
            maxit=1000,
            rel_tol=1e-5,
            abs_tol=1e-3,
        )

        min_chi2_values.append(min_chi2)
        best_params_chi2.append(p_opt)

        ax = axs[datafiles.index(datafile)]
        # Plot the data and the best-fit model for each data file in a subplot.
        # ax.hist(# plot the histogram of the data
        #     radius, bins=bins, weights=np.ones_like(radius)*Nsat
        # )
        ax.stairs(Ntilde_data, edges=bin_edges, label="binned_data")

        # define a function that computes the predicted number of satellites in a bin as
        # given by the parameters that minimize chi2
        def model(bin, a, b, c):
            func = lambda x: satellite_number(
                x=x,
                A=get_normalization_constant(a=a, b=b, c=c),
                Nsat=Nsat,
                a=a,
                b=b,
                c=c,
            )
            # integrate the number counts over the bin, set order for computational speed.
            I = romberg_integrator(func, (bin_edges[bin], bin_edges[bin + 1]), order=5)
            return I / bin_widths[bin]

        Ntilde_model = [model(bin, *p_opt) for bin in np.arange(nbins)]

        ax.stairs(  # bin the best-fit model using the best-fit parameters found from chi-squared minimization
            Ntilde_model, bin_edges, label="binned_fit"
        )

        x_plot = np.geomspace(x_lower, x_upper, 100)
        # plot the best-fit model using the best-fit parameters found from chi-squared minimization
        ax.plot(
            x_plot,
            satellite_number(x_plot, get_normalization_constant(*p_opt), Nsat, *p_opt),
            label="x*N(x)",
        )

        # Add labels and title to the subplot
        ax.set_title(f"Data file: {datafile}")
        ax.set_xlabel("x = r / r_virial")
        ax.set_ylabel("Number of satellites")

        # log-log scaling
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(None, None)
        ax.legend()
        plt.savefig("Plots/satellite_fits_chi2.png")

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


best_params_poisson = []


def do_question_1c():
    # ======== Question 1c: Fitting N(x) with Poisson ln-likelihood ========
    datafiles = ["m11", "m12", "m13", "m14", "m15"]

    min_poisson_llh_values = []

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
        func = lambda abc: g(*abc, radius)
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
        ax.stairs(Ntilde_data, edges=bin_edges, label="binned_data")
        ax.plot(  # plot the model that maximizes Poisson likelihood
            x_plot,
            satellite_number(x_plot, get_normalization_constant(*p_opt), Nsat, *p_opt),
        )

        # Add labels and title to the subplot
        ax.set_title(f"Data file: {datafile}")
        ax.set_xlabel("x = r / r_virial")
        ax.set_ylabel("Number of satellites")

        # log-log scaling
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, None)

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


def do_question_1d():
    # ======== Question 1d: Statistical tests ========
    datafiles = ["m11", "m12", "m13", "m14", "m15"]

    G_scores_chi2 = []
    Q_scores_chi2 = []

    G_scores_poisson = []
    Q_scores_poisson = []

    for datafile in datafiles:
        radius, nhalo = readfile(f"Data/satgals_{datafile}.txt")

        # Use best-fit parameters from previous steps
        p_chi2 = best_params_chi2[datafiles.index(datafile)]
        p_poisson = best_params_poisson[datafiles.index(datafile)]

        # construct bins
        nbins = 10
        x_lower, x_upper = (
            np.min(radius),
            np.max(radius),
        )
        bin_edges = np.geomspace(x_lower, x_upper, nbins + 1)
        bin_widths = bin_edges[1:] - bin_edges[:-1]

        # compute the binned data
        bin_counts = np.histogram(radius, bin_edges)[
            0
        ]  # "mean number of satellites in each radial bin"
        data = bin_counts / (nhalo * bin_widths)
        Nsat = np.sum(bin_counts) / nhalo

        # generate expected data from chi2 fit:
        def model(bin, a, b, c):
            func = lambda x: satellite_number(
                x=x,
                A=get_normalization_constant(a=a, b=b, c=c),
                Nsat=Nsat,
                a=a,
                b=b,
                c=c,
            )
            # integrate the number counts over the bin, set order for computational speed.
            I = romberg_integrator(func, (bin_edges[bin], bin_edges[bin + 1]), order=5)
            return I / bin_widths[bin]

        chi2_data = [model(bin, *p_chi2) for bin in np.arange(nbins)]
        G_score_chi2 = Gtest(data, chi2_data)

        # generate expected data from poisson fit:
        def model(bin, a, b, c):
            func = lambda x: satellite_number(
                x=x,
                A=get_normalization_constant(a=a, b=b, c=c),
                Nsat=Nsat,
                a=a,
                b=b,
                c=c,
            )
            # integrate the number counts over the bin, set order for computational speed.
            I = romberg_integrator(func, (bin_edges[bin], bin_edges[bin + 1]), order=5)
            return I / bin_widths[bin]

        chi2_data = [model(bin, *p_poisson) for bin in np.arange(nbins)]
        G_score_poisson = Gtest(data, chi2_data)

        # Append the G and Q scores for chi2 and poisson fits to their respective arrays
        G_scores_chi2.append(G_score_chi2)
        Q_scores_chi2.append(0.0)
        G_scores_poisson.append(G_score_poisson)
        Q_scores_poisson.append(0.0)

    # Save G and Q scores for chi2 and poisson fits to tex files for later use in the PDF
    with open("Calculations/statistical_test_table_rows.tex", "w") as f:
        rows = []
        for i, (G, Q) in enumerate(zip(G_scores_chi2, Q_scores_chi2), start=11):
            rows.append(f"$\\chi^2$ & m{i} & {G:.5f} & {Q:.5f}")

        for i, (G, Q) in enumerate(zip(G_scores_poisson, Q_scores_poisson), start=11):
            rows.append(f"Poisson & m{i} & {G:.5f} & {Q:.5f}")

        for idx, row in enumerate(rows):
            if idx < len(rows) - 1:
                f.write(row + " \\\\\n")
            else:
                f.write(row)


def do_question_1e():
    # ======== Question 1e: Monte Carlo simulations ========
    # pick one of the data files to perform the Monte Carlo simulations on, e.g. m12
    datafiles = ["m11", "m12", "m13", "m14", "m15"]
    index = (
        1  # index of the data file to use for Monte Carlo simulations, e.g. 1 for m12
    )

    radius, nhalo = readfile(f"Data/satgals_{datafiles[index]}.txt")

    # Use best-fit parameters from previous steps for the original data file
    best_params_chi2 = (0.0, 0.0, 0.0)  # replace by the correct array
    best_params_poisson = (0.0, 0.0, 0.0)  # replace by the correct array

    pseudo_chi2_params = []
    pseudo_poisson_params = []

    num_pseudo_experiments = 10  # replace by number with reasonable runtime
    for i in range(num_pseudo_experiments):

        # TODO: generate pseudo-data by sampling from original best-fit chi2 and poisson models
        # Then, for each pseudo-dataset, perform the chi2 and poisson fits to find the best-fit parameters.

        # Append the best-fit parameters for each pseudo-dataset to their respective arrays.
        pseudo_chi2_params.append(
            (0.0, 0.0, 0.0)
        )  # replace by the correct best-fit parameters (a,b,c) found from chi-squared minimization for the pseudo-dataset
        pseudo_poisson_params.append(
            (0.0, 0.0, 0.0)
        )  # replace by the correct best-fit parameters (a,b,c) found from Poisson negative log-likelihood minimization for the pseudo-dataset

    # plot the pseudo best-fit profiles, plot the original best-fit profile in another color and plot the mean in one more color

    # chi2 plot
    x_plot = np.linspace(1e-4, 5, 100)  # create x_array for plotting the model
    plt.figure(figsize=(6.4, 4.8))
    for params in pseudo_chi2_params:
        plt.plot(
            x_plot, np.ones_like(x_plot)
        )  # plot the best-fit model for each pseudo-dataset using the best-fit parameters found from chi-squared minimization

    plt.plot(
        x_plot, np.ones_like(x_plot)
    )  # plot the original best-fit model using the best-fit parameters found from chi-squared minimization on the real data

    mean_params_chi2 = np.mean(
        pseudo_chi2_params, axis=0
    )  # calculate the mean of the best-fit parameters from the pseudo-datasets
    plt.plot(
        x_plot, np.ones_like(x_plot)
    )  # plot the mean of the best-fit models from the pseudo-datasets

    plt.title(f"Monte Carlo simulations - chi2 fit - Data file: {datafiles[index]}")
    plt.xlabel("x = r / r_virial")
    plt.ylabel("Number of satellites")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(["Pseudo-dataset fits", "Original fit", "Mean of pseudo fits"])
    plt.savefig("Plots/satellite_monte_carlo_chi2.png")

    # poisson plot
    x_plot = np.linspace(1e-4, 5, 100)  # create x_array for plotting the model
    plt.figure(figsize=(6.4, 4.8))
    for params in pseudo_poisson_params:
        plt.plot(
            x_plot, np.ones_like(x_plot)
        )  # plot the best-fit model for each pseudo-dataset using the best-fit parameters found from Poisson negative log-likelihood minimization
    plt.plot(
        x_plot, np.ones_like(x_plot)
    )  # plot the original best-fit model using the best-fit parameters found from Poisson negative log-likelihood minimization on the real data

    mean_params_poisson = np.mean(
        pseudo_poisson_params, axis=0
    )  # calculate the mean of the best-fit parameters from the pseudo-datasets
    plt.plot(
        x_plot, np.ones_like(x_plot)
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
