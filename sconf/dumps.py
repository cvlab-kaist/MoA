from .utils import colorize, kv_iter


def dump_args(args):
    """ Convert args (argparse.Namespace) to printable string """
    lines = []
    key_len = max(map(len, vars(args).keys()))
    for attr, value in vars(args).items():
        line = "{:{key_len}s} = {}".format(attr, value, key_len=key_len)
        lines.append(line)

    return "\n".join(lines)


def dump_config(config, modified_color=36, quote_str=False):
    """ Dump to colorized string

    Args:
        config (Config): config to dump
        modified_color (int): color for modified item. Use ``None`` for non-coloring.
        quote_str (bool): quoting string for identifying string with keyword. Default: ``False``
    """
    strs = []
    tab = '  '

    def quote(v):
        if quote_str and isinstance(v, str) and v:
            return "'{}'".format(v)
        return v

    def repr_dict(k, v):
        if isinstance(v, (dict, list)):
            v = ''
        return "{}: {}\n".format(k, quote(v))

    def repr_list(_k, v):
        if isinstance(v, (dict, list)):
            return "- "
        return "- {}\n".format(quote(v))

    def dump(data, indent, last, firstline_nopref=False):
        modified = config._sconf_modified.get(id(data), set())

        N = len(data)
        for i, (k, v) in enumerate(kv_iter(data)):
            prefix = indent
            if firstline_nopref:
                prefix = ''
                firstline_nopref = False

            # indentation is determined by parent data type
            if isinstance(data, dict):
                s = repr_dict(k, v)
                skip_first_indent = False
            elif isinstance(data, list):
                s = repr_list(k, v)
                skip_first_indent = True
            else:
                raise ValueError(type(data))

            is_last = last and i == N-1
            if is_last and not isinstance(v, (dict, list)):
                s = s.rstrip('\n')

            if k in modified:
                s = colorize(s, modified_color)
            strs.append(prefix + s)
            if isinstance(v, (dict, list)):
                dump(v, indent + tab, is_last, skip_first_indent)

    dump(config.get_data(), '', True)
    return ''.join(strs)
