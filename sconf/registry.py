REGISTRY = {}

def has_key(key):
    return key in REGISTRY


def register(cfg, key, ignore_duplicated_error=False):
    """Register cfg (Config) by key"""
    if has_key(key):
        if ignore_duplicated_error:
            return
        else:
            raise ValueError(f"{key} already exists in registry")

    REGISTRY[key] = cfg


def get(key):
    return REGISTRY[key]


def reset():
    global REGISTRY
    REGISTRY = {}
