from collections import Iterable

def flatten(lst):
    for x in lst:
        if isinstance(x, list):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x

def change_interval (x, inmin, inmax, outmin, outmax):
    # normalizing x between 0 and 1
    x = (x - inmin) / (inmax - inmin)
    # denormalizing between outmin and outmax
    return x * (outmax - outmin) + outmin