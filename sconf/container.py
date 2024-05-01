import copy
import io
from pathlib import Path

import munch
from ruamel.yaml import YAML
from ruamel.yaml.scalarbool import ScalarBoolean
from ruamel.yaml.scalarfloat import ScalarFloat
from ruamel.yaml.scalarint import ScalarInt
from ruamel.yaml.scalarstring import ScalarString


class DictContainer:
    """ Dictionary container """
    _yaml = YAML()

    def __init__(self, data=None):
        self.set_data_from_key(data or {})

    def get(self, *args, **kwargs):
        return self._sconf_data.get(*args, **kwargs)

    def pop(self, *args, **kwargs):
        return self._sconf_data.pop(*args, **kwargs)

    def asdict(self):
        dic = self._sconf_data.toDict()
        def cvt_type(dic):
            for k, v in dic.items():
                if isinstance(v, dict):
                    dic[k] = cvt_type(v)
                elif isinstance(v, ScalarFloat):
                    dic[k] = float(v)
                elif isinstance(v, ScalarBoolean):
                    dic[k] = bool(v)
                elif isinstance(v, ScalarInt):
                    dic[k] = int(v)
                elif isinstance(v, ScalarString):
                    dic[k] = str(v)

            return dic

        return cvt_type(dic)

    def __str__(self):
        return str(self.asdict())

    def __repr__(self):
        return repr(self.asdict())

    def __getitem__(self, key):
        return self._sconf_data[key]

    def __setitem__(self, key, value):
        self._sconf_data[key] = value

    def __getattr__(self, key):
        return self._sconf_data[key]

    def __setattr__(self, key, value):
        if key.startswith('_sconf_'):
            super().__setattr__(key, value)
        else:
            self._sconf_data[key] = value

    def __contains__(self, key):
        return key in self._sconf_data

    def __len__(self):
        return len(self._sconf_data)

    def __eq__(self, other):
        return self._sconf_data == other

    def items(self):
        return self._sconf_data.items()

    def keys(self):
        return self._sconf_data.keys()

    def values(self):
        return self._sconf_data.values()

    def get_data(self):
        return self._sconf_data

    def set_data_from_key(self, key):
        """ Set data from key

        Args:
            key (dict, str, pathlib.Path, io.IOBase): data dictionary, data path,
                or IOBase (file pointer)
        """
        data = self._load_key(key)
        self._sconf_data = data

    def _load_key(self, key):
        """ Load data from key

        Args:
            key (dict, str, pathlib.Path, io.IOBase): data dictionary, data path,
                or IOBase (file pointer)
        """
        if isinstance(key, dict):
            dic = copy.deepcopy(key)
        elif isinstance(key, (str, Path)):
            dic = self._yaml.load(open(key, encoding="utf-8"))
        elif isinstance(key, io.IOBase):
            dic = self._yaml.load(key)
        else:
            raise ValueError(
                "Key should be dict, str, or pathlib.Path, but {} is given".format(type(key))
            )

        return munch.Munch.fromDict(dic)

    def __getstate__(self):
        return copy.deepcopy(self.__dict__)

    def __setstate__(self, state):
        self.__dict__.update(state)

    #  def __copy__(self):
    #      new = type(self)()
    #      new.__dict__.update(self.__dict__)
    #      return new

    def __deepcopy__(self, memo):
        new = type(self)()
        new.__dict__.update(copy.deepcopy(self.__dict__, memo))
        return new
