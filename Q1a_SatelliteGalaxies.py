# imports
import numpy as np
import matplotlib.pyplot as plt

from minimizer import Minimizer


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


def do_question_1a():
    # ======== Question 1a: Maximization of N(x) ========
    a = 2.4
    b = 0.25
    c = 1.6
    Nsat = 100
    A_1a = 256 / (5 * np.pi ** (3 / 2))
    # x_lower, x_upper = 10**-4, 5

    # set N in logspace to make computations faster, use -N since were finding maximum with minimizer.
    negative_N_logspace = lambda logx: -satellite_number(
        np.exp(logx), A_1a, Nsat, a, b, c
    )

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


if __name__ == "__main__":
    do_question_1a()
