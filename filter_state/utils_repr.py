import math


def round_to_2_sigfigs_scientific(x):
    if x == 0:
        return 0.0
    exponent = math.floor(math.log10(abs(x)))
    factor = 10 ** (exponent - 1)
    return round(x / factor) * factor
