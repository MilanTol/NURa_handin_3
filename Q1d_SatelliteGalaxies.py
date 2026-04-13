# imports
import numpy as np
import matplotlib.pyplot as plt

from romberg_integrator import romberg_integrator
from scipy.special import gammainc


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


def Qscore(k: int, x: float) -> float:
    """
    computes probability of finding a chi^2 at least
    as bad as x by chance, given k degrees of freedom.

    Args:
        k (int): degrees of freedom
        x (float): minimum chi2 value

    Returns:
        float: Qscore
    """
    # we are allowed to use special library functions
    # to compute the gamma functions.
    # scipy.special.gammainc computes P(x,k) directly.
    Pkx = gammainc(0.5 * k, 0.5 * x)
    return 1 - Pkx


def do_question_1d():
    # ======== Question 1d: Statistical tests ========
    datafiles = ["m11", "m12", "m13", "m14", "m15"]

    G_scores_chi2 = []
    Q_scores_chi2 = []

    G_scores_poisson = []
    Q_scores_poisson = []

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

    for datafile in datafiles:
        radius, nhalo = readfile(f"Data/satgals_{datafile}.txt")

        # load best fit parameters for current datafile
        p_chi2 = best_params_chi2[datafiles.index(datafile)]
        p_poisson = best_params_poisson[datafiles.index(datafile)]

        # construct bins
        nbins = 20
        x_lower, x_upper = (
            0.5 * np.min(radius),
            2 * np.max(radius),
        )
        bin_edges = np.geomspace(x_lower, x_upper, nbins + 1)

        # compute the binned data
        bin_counts = np.histogram(radius, bin_edges)[
            0
        ]  # "mean number of satellites in each radial bin"
        data = bin_counts
        Nsat = np.sum(bin_counts)

        # define the model
        def model(bin, a, b, c):
            func = lambda x: satellite_number(
                x=x,
                A=get_normalization_constant(a=a, b=b, c=c, order=7),
                Nsat=Nsat,
                a=a,
                b=b,
                c=c,
            )
            # integrate the number counts over the bin, set order for computational speed.
            I = romberg_integrator(func, (bin_edges[bin], bin_edges[bin + 1]), order=5)
            return I

        # G-test requires that sum_i O_i = sum_i E_i:
        # the total of the two data set counts must be equal.

        # generate expected data from chi2 fit:
        chi2_data = np.array([model(bin, *p_chi2) for bin in np.arange(nbins)])
        E_tot = np.sum(chi2_data)
        O_tot = np.sum(data)
        chi2_data *= O_tot / E_tot

        G_score_chi2 = Gtest(data, chi2_data)
        G_scores_chi2.append(G_score_chi2)

        # generate expected data from poisson fit:
        poisson_data = np.array([model(bin, *p_poisson) for bin in np.arange(nbins)])
        E_tot = np.sum(poisson_data)
        O_tot = np.sum(data)
        poisson_data *= O_tot / E_tot

        G_score_poisson = Gtest(data, poisson_data)
        G_scores_poisson.append(G_score_poisson)

        # the degrees of freedom is given by
        # k = N - M
        # where
        #   N is the number of data points: amount of bins
        #   M is the number of independent parameters: 3
        k = nbins - 3

        # we compute the Qscores by inputting Gscores:
        Q_score_chi2 = Qscore(k, G_score_chi2)
        Q_scores_chi2.append(Q_score_chi2)
        Q_score_poisson = Qscore(k, G_score_poisson)
        Q_scores_poisson.append(Q_score_poisson)

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


if __name__ == "__main__":
    do_question_1d()
