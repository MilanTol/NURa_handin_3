import numpy as np


class RNG:
    def __init__(self, seed=1):
        if seed == 0:
            raise KeyError("seed must be greater than 0")
        self.xor1 = np.uint64(seed)
        self.xor2 = np.uint64(seed) ^ self.XOR1()  # uncorrelate the seeds
        self.mwc = np.uint64(seed) ^ self.XOR2()

    def XOR1(self):
        self.xor1 ^= self.xor1 >> np.uint64(21)
        self.xor1 ^= self.xor1 << np.uint64(35)
        self.xor1 ^= self.xor1 >> np.uint64(4)
        return self.xor1

    def XOR2(
        self,
    ):  # from some reddit posts I saw this is another common choice: 13, 17, 5
        self.xor2 ^= self.xor2 >> np.uint64(13)
        self.xor2 ^= self.xor2 << np.uint64(17)
        self.xor2 ^= self.xor2 >> np.uint64(5)
        return self.xor2

    def MWC(self):
        a = np.uint64(4294957665)
        self.mwc = a * (self.mwc & np.uint64(2**32 - 1)) + (self.mwc >> np.uint64(32))
        return np.uint32(self.mwc)  # use uint32 to only return the lower 32 bits

    def int(self, bounds: tuple = None) -> np.uint64:
        """
        random number generator

        Parameters
        ----------
        bounds (optional, tuple): contains integers for the lower bound and upper bound

        Returns
        -------
        a random np.int64 in between the two bounds if provided (including bounds).
        Otherwise in range of np.uint64.
        """

        # using python int to prevent overflow warnings
        x = int(self.XOR1() + self.XOR2())  # add the two XOR methods
        x ^= self.MWC()  # or operation with MWC for 'extra' randomness

        if bounds is None:
            return np.uint64(x)  # do return number as uint64
        else:
            a, b = bounds
            x = x % np.int64(b - a + 1)
            return np.int64(x + a)

    def float(self, bounds: tuple = None) -> np.float64:
        """
        random number generator

        Parameters
        ----------
        bounds (optional, tuple): contains numbers for the lower bound and upper bound

        Returns
        -------
        a random np.float64 in between the two bounds if provided. Otherwise in range of (0, 1).
        """

        if bounds is None:
            return (
                self.int() / 2**64
            )  # np.int64 can obtain integers in the range [0, 2^64]
        else:
            a, b = bounds
            return (self.int() / 2**64) * (b - a) + a
