def round_to_odd(num):
    """
    Rounds a number to the nearest odd integer.
    """
    r = round(num, 0)
    if r % 2 == 0:
        if (num - r) != 0:
            return int(r + ((num - r) / abs(num - r)))
        else:
            return int(r - 1)
    else:
        return int(r)
