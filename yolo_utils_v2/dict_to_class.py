
from collections import namedtuple


def dict_to_class(dict):
    DictClass = namedtuple('dict_class', dict.keys())
    dict_class = DictClass(*dict.values())
    return dict_class
