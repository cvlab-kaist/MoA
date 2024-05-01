def colorize(s, color):
    if color is None:
        return s

    return "\033[{}m{}\033[0m".format(color, s)


def kv_iter(ds):
    """ Iterator for key-value structure, list and dict """
    if isinstance(ds, list):
        return enumerate(ds)
    elif isinstance(ds, dict):
        return iter(ds.items())
    else:
        raise ValueError(type(ds))
