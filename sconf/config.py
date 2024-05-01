import sys

from . import registry
from .container import DictContainer
from .dumps import dump_config
from .types import type_infer
from .utils import kv_iter


class Config(DictContainer):
    """ Config container """
    def __init__(self, *keys, default=None, colorize_modified_item=True, registry_key='default'):
        """
        Args:
            keys (str, dict, or fp ...): yaml path, loaded dict, or file pointer
            default (str or dict): default key. Default: ``None``
            colorize_modified_item (bool): a flag for coloring modified items on dumps().
                Default: ``True``
            registry_key (str): every Config object is automatically registered
        """
        super().__init__()
        self._sconf_colorize_modified_item = colorize_modified_item
        self._sconf_modified = {}
        self._sconf_keydic = {}

        if default:
            keys = (default,) + keys

        if keys:
            self.set_data_from_key(keys[0])
            keys = keys[1:]

        for key in keys:
            self._dict_update(self._load_key(key))

        self._build_keydic()

        ignore_duplicated_error = registry_key == 'default'
        registry.register(self, registry_key, ignore_duplicated_error)

    @staticmethod
    def from_registry(key):
        return registry.get(key)

    @staticmethod
    def get_default():
        return Config.from_registry('default')

    def _dict_update(self, dic):
        """ update data from dic - support nested dic """
        def merge(base, supp):
            """ Merge supplementary dict into base dict """
            for k in supp.keys():
                if isinstance(supp[k], dict) and k in base:
                    merge(base[k], supp[k])
                else:
                    base[k] = supp[k]

        if dic is not None:
            merge(self.get_data(), dic)

    def _build_keydic(self):
        """ Build key dictionary; keydic[flat_key] = lastdic """
        def build_keydic(data, prefix, keydic):
            for k, v in kv_iter(data):
                key = "{}.{}".format(prefix, k)
                keydic[key] = data
                if isinstance(v, (dict, list)):
                    build_keydic(v, key, keydic)

        self._sconf_keydic = {}  # reset
        build_keydic(self.get_data(), '', self._sconf_keydic)

    def argv_update(self, argv=None):
        """ Update data using argv
        Argument key has two key types, "--" and "---".
        The double-dash key "--" is used to modify a single item, and
        the triple-dash key "---" is used to modify multiple items at once.

        Args:
            argv (list): argument list; [key1, value1, key2, value2, ...]
                Can be set to ``None`` to use sys.argv[1:]. Default: ``None``
        """
        if argv is None:
            argv = sys.argv[1:]

        N = len(argv)
        if N % 2 != 0:
            raise ValueError("Key-value should be paired, but given argv = {}".format(argv))

        for i in range(0, N, 2):
            flat_key, value = argv[i:i+2]
            self._update(flat_key, value)

        self._build_keydic()

    def _update(self, flat_key, value):
        """ Update data using flat_key and value

        Args:
            flat_key (str): hierarchical (partial) flat key with:
                "--": must single-match
                "---": allow multi-match
                e.g.) "--model.n_layers" or "---self_attention"
            value (str)
        """
        index = 0
        while flat_key[index] == '-':
            index += 1

        if index not in {2, 3}:
            raise ValueError("Key should have `--` or `---` prefix, but {}".format(flat_key))

        flat_key = flat_key[index:]

        lasts = self._find_lastdic(flat_key)
        key = flat_key.split('.')[-1]

        # single match case
        if index == 2:
            if len(lasts) == 0:
                raise ValueError(f"key `{flat_key}` do not match to any keys")

            if len(lasts) >= 2:
                raise ValueError(
                    "key `{}` matches too many keys (#{})".format(flat_key, len(lasts))
                )

        for last in lasts:
            if isinstance(last, list):
                key = int(key)
                #  if key == len(last):
                #      last.append(None)  # extend list (new arg)
            last[key] = type_infer(value)

            if self._sconf_colorize_modified_item:
                self._sconf_modified.setdefault(id(last), set()).add(key)

    def _find_lastdic(self, flat_key):
        """ Find parent dictionary of given flat_key """
        def get_parentkey(key):
            return key[:key.rindex('.')]

        key = '.' + flat_key
        key_parent = get_parentkey(key)
        matches = []
        candidates = {}
        for k, v in self._sconf_keydic.items():
            k_parent = get_parentkey(k)
            if k.endswith(key):
                matches.append(v)
            elif k_parent.endswith(key_parent):
                candidates[k_parent] = v

        # add new argument
        #  if not matches and len(candidates) == 1:
        #      return [candidates.popitem()[1]]

        return matches

    def dumps(self, modified_color=36, quote_str=False):
        """ Dump to colorized string

        Args:
            modified_color (int): color for modified item. Can be set to ``None`` for non-coloring
            quote_str (bool): quoting string for identifying string with keyword. Default: ``False``
        """
        return dump_config(self, modified_color, quote_str)
