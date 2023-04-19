
def strtobool(val):
    """
    Silly re-implementation of strtobool, since python has
    deprecated these things...

    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError("Invalid bool value %r" % (val,))


